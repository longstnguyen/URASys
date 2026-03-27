import json
import os
import asyncio
import pathlib
import traceback
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from google import genai
from google.genai import types

# Derive project root from this file's location
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent

# Ensure env file is loaded from the correct absolute path
if not os.environ.get("ENVIRONMENT_FILE"):
    os.environ["ENVIRONMENT_FILE"] = str(PROJECT_ROOT / "environments" / ".env")

MODELS_CONFIG_PATH = PROJECT_ROOT / "src" / "urasys" / "config" / "models_config.json"
FAQ_BM25_PATH = PROJECT_ROOT / "src" / "urasys" / "data" / "bm25" / "faq" / "state_dict.json"
DOC_BM25_PATH = PROJECT_ROOT / "src" / "urasys" / "data" / "bm25" / "document" / "state_dict.json"

from copilotkit import CopilotKitRemoteEndpoint, Action
from copilotkit.integrations.fastapi import add_fastapi_endpoint

from urasys.config.system_config import SETTINGS
from urasys.config.utils import get_milvus_config
from urasys.core.model_clients.bm25 import BM25Client
from urasys.core.model_clients.embedder.openai import OpenAIClientConfig, OpenAIEmbedder
from urasys.core.retriever.faq_retriever import FAQRetriever
from urasys.core.retriever.document_retriever import DocumentRetriever
from urasys.utils.base_class import ModelsConfig
from urasys.utils.database_clients.milvus import MilvusVectorDatabase, MilvusConfig
from urasys.prompts.query import FAQ_SEARCH_INSTRUCTION_PROMPT, DOCUMENT_SEARCH_INSTRUCTION_PROMPT

faq_retriever: FAQRetriever = None
doc_retriever: DocumentRetriever = None
_gemini_client: genai.Client = None
_retrieval_model_id: str = "gemini-2.5-flash"
_SUB_AGENT_MAX_RETRIES: int = 3


def _init_retrievers() -> None:
    global faq_retriever, doc_retriever, _gemini_client, _retrieval_model_id
    with open(MODELS_CONFIG_PATH) as f:
        models_config = json.load(f)
    embedding_cfg = ModelsConfig.from_dict(models_config).embedding_config
    embedder = OpenAIEmbedder(config=OpenAIClientConfig(
        api_key=SETTINGS.OPENAI_API_KEY,
        model=embedding_cfg.model_id,
    ))
    vector_db = MilvusVectorDatabase(config=get_milvus_config(SETTINGS))
    if not FAQ_BM25_PATH.exists() or not DOC_BM25_PATH.exists():
        raise FileNotFoundError(
            "BM25 state dicts not found. Run `python scripts/index_offline.py` first."
        )
    faq_bm25 = BM25Client(local_path=str(FAQ_BM25_PATH), init_without_load=False)
    doc_bm25 = BM25Client(local_path=str(DOC_BM25_PATH), init_without_load=False)
    faq_retriever = FAQRetriever(
        collection_name=SETTINGS.MILVUS_COLLECTION_FAQ_NAME,
        embedder=embedder,
        bm25_client=faq_bm25,
        vector_db=vector_db,
    )
    doc_retriever = DocumentRetriever(
        collection_name=SETTINGS.MILVUS_COLLECTION_DOCUMENT_NAME,
        embedder=embedder,
        bm25_client=doc_bm25,
        vector_db=vector_db,
    )
    _gemini_client = genai.Client(api_key=SETTINGS.GEMINI_API_KEY)
    cfg = ModelsConfig.from_dict(models_config)
    if hasattr(cfg, "llm_config") and "retrieval_llm" in cfg.llm_config:
        _retrieval_model_id = cfg.llm_config["retrieval_llm"].model_id
    logger.info(f"CopilotKit backend: retrievers initialized, sub-agent LLM model={_retrieval_model_id}.")


try:
    _init_retrievers()
except Exception as e:
    logger.error(f"Failed to initialize retrievers: {e}")
    logger.error(traceback.format_exc())


def search_faqs(query: str) -> str:
    if faq_retriever is None:
        return "FAQ retriever is not available. Please index your data first."
    try:
        results = faq_retriever.retrieve_faqs(query, top_k=5)
        if not results.faqs:
            return "No relevant FAQs found for this query."
        lines = []
        for i, faq in enumerate(results.faqs, 1):
            lines.append(f"[FAQ {i}]")
            lines.append(f"Q: {faq.source_node.question}")
            lines.append(f"A: {faq.source_node.answer}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"search_faqs error: {e}")
        return f"Search failed: {e}"


def search_documents(query: str) -> str:
    if doc_retriever is None:
        return "Document retriever is not available. Please index your data first."
    try:
        results = doc_retriever.retrieve_documents(query, top_k=5)
        if not results.documents:
            return "No relevant document passages found for this query."
        lines = []
        for i, doc in enumerate(results.documents, 1):
            lines.append(f"[Passage {i}]")
            lines.append(doc.source_node.chunk)
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"search_documents error: {e}")
        return f"Search failed: {e}"


# ─────────────────────────────────────────────────────────────────
# Sub-agent tool declarations (google-genai format)
# ─────────────────────────────────────────────────────────────────

_FAQ_TOOL = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="faq_retrieval_tool",
        description="Retrieve top K relevant FAQs based on the query.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(type=types.Type.STRING, description="The search query."),
                "top_k": types.Schema(type=types.Type.INTEGER, description="Number of FAQs to retrieve (default 5)."),
            },
            required=["query"],
        ),
    )
])

_DOC_TOOL = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="document_retrieval_tool",
        description="Retrieve top K relevant document passages based on the query.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(type=types.Type.STRING, description="The search query."),
                "top_k": types.Schema(type=types.Type.INTEGER, description="Number of document passages to retrieve (default 3)."),
            },
            required=["query"],
        ),
    )
])


# ─────────────────────────────────────────────────────────────────
# Raw retrieval helpers – return (formatted_text, raw_list)
# ─────────────────────────────────────────────────────────────────

def _sync_retrieve_faqs(query: str, top_k: int = 5) -> tuple:
    """Sync wrapper. Returns (formatted_text, raw_list)."""
    if faq_retriever is None:
        return "FAQ retriever is not available.", []
    try:
        results = faq_retriever.retrieve_faqs(query, top_k=top_k)
        if not results.faqs:
            return "No relevant FAQs found.", []
        lines, raw = [], []
        for i, faq in enumerate(results.faqs, 1):
            lines.append(f"[FAQ {i}] Q: {faq.source_node.question}\nA: {faq.source_node.answer}")
            raw.append({
                "question": faq.source_node.question,
                "answer": faq.source_node.answer,
                "score": round(float(faq.score), 4) if hasattr(faq, "score") else None,
            })
        return "\n\n".join(lines), raw
    except Exception as e:
        logger.error(f"_sync_retrieve_faqs error: {e}")
        return f"Search failed: {e}", []


def _sync_retrieve_docs(query: str, top_k: int = 3) -> tuple:
    """Sync wrapper. Returns (formatted_text, raw_list)."""
    if doc_retriever is None:
        return "Document retriever is not available.", []
    try:
        results = doc_retriever.retrieve_documents(query, top_k=top_k)
        if not results.documents:
            return "No relevant document passages found.", []
        lines, raw = [], []
        for i, doc in enumerate(results.documents, 1):
            lines.append(f"[Passage {i}]\n{doc.source_node.chunk}")
            raw.append({
                "chunk": doc.source_node.chunk,
                "score": round(float(doc.score), 4) if hasattr(doc, "score") else None,
            })
        return "\n\n".join(lines), raw
    except Exception as e:
        logger.error(f"_sync_retrieve_docs error: {e}")
        return f"Search failed: {e}", []


# ─────────────────────────────────────────────────────────────────
# Sub-agent LLM loops (paper Section 3.2 / Algorithm 1)
# ─────────────────────────────────────────────────────────────────

async def _run_faq_sub_agent(
    user_query: str,
    event_queue: asyncio.Queue | None = None,
    query_idx: int = 0,
) -> dict:
    """
    FAQ Search Agent: iterative LLM loop with faq_retrieval_tool.
    Follows paper Figure fig:prompt_faq — manages up to max_retries tool-call
    attempts, reformulating the query on each unsuccessful round.

    Returns: {answer: str, faq_results: list, attempts: int, steps: list}
    """
    loop = asyncio.get_running_loop()

    # Fallback: no LLM available – call retriever directly
    if _gemini_client is None:
        text, raw = await loop.run_in_executor(None, lambda: _sync_retrieve_faqs(user_query))
        return {"answer": text, "faq_results": raw, "attempts": 1, "steps": []}

    system_prompt = FAQ_SEARCH_INSTRUCTION_PROMPT.format(max_retries=_SUB_AGENT_MAX_RETRIES)
    contents: list = [types.Content(role="user", parts=[types.Part(text=user_query)])]
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[_FAQ_TOOL],
        temperature=0.1,
    )

    all_raw_faqs: list = []
    steps: list = []
    attempts = 0

    for attempt in range(1, _SUB_AGENT_MAX_RETRIES + 1):
        attempts = attempt
        try:
            # Run synchronous Gemini call off the event loop thread, with 30s timeout
            captured = list(contents)
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: _gemini_client.models.generate_content(
                        model=_retrieval_model_id,
                        contents=captured,
                        config=config,
                    ),
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"_run_faq_sub_agent LLM call timed out (attempt {attempt})")
            break
        except Exception as e:
            logger.error(f"_run_faq_sub_agent LLM call failed (attempt {attempt}): {e}")
            break

        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.content is None:
            break

        # Extract reasoning text the LLM produced in this turn
        reasoning_text = " ".join(
            p.text for p in (candidate.content.parts or []) if hasattr(p, "text") and p.text
        ).strip()

        tool_calls = [p for p in (candidate.content.parts or []) if p.function_call]
        if not tool_calls:
            # LLM produced its final grounded answer
            return {
                "answer": reasoning_text or "No relevant FAQ found for the current request.",
                "faq_results": all_raw_faqs,
                "attempts": attempts,
                "steps": steps,
            }

        # Append model turn to conversation
        contents.append(candidate.content)

        # Execute every tool call the model requested
        tool_response_parts = []
        for part in tool_calls:
            fc = part.function_call
            q = fc.args.get("query", user_query)
            k = int(fc.args.get("top_k", 5))
            text, raw = await loop.run_in_executor(None, lambda qq=q, kk=k: _sync_retrieve_faqs(qq, kk))
            all_raw_faqs.extend(raw)
            tool_response_parts.append(
                types.Part(function_response=types.FunctionResponse(name=fc.name, response={"result": text}))
            )
            step = {"round": attempt, "tool_query": q, "num_results": len(raw), "reasoning": reasoning_text}
            steps.append(step)
            if event_queue is not None:
                await event_queue.put({"type": "round", "query_idx": query_idx, "agent": "faq", **step})
        contents.append(types.Content(role="user", parts=tool_response_parts))

    return {
        "answer": "No relevant FAQ found for the current request.",
        "faq_results": all_raw_faqs,
        "attempts": attempts,
        "steps": steps,
    }


async def _run_doc_sub_agent(
    user_query: str,
    event_queue: asyncio.Queue | None = None,
    query_idx: int = 0,
) -> dict:
    """
    Document Search Agent: iterative LLM loop with document_retrieval_tool.
    Follows paper Figure fig:prompt_document — same retry logic as FAQ agent.

    Returns: {answer: str, doc_results: list, attempts: int}
    """
    loop = asyncio.get_running_loop()

    if _gemini_client is None:
        text, raw = await loop.run_in_executor(None, lambda: _sync_retrieve_docs(user_query))
        return {"answer": text, "doc_results": raw, "attempts": 1, "steps": []}

    system_prompt = DOCUMENT_SEARCH_INSTRUCTION_PROMPT.format(max_retries=_SUB_AGENT_MAX_RETRIES)
    contents: list = [types.Content(role="user", parts=[types.Part(text=user_query)])]
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[_DOC_TOOL],
        temperature=0.1,
    )

    all_raw_docs: list = []
    steps: list = []
    attempts = 0

    for attempt in range(1, _SUB_AGENT_MAX_RETRIES + 1):
        attempts = attempt
        try:
            captured = list(contents)
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: _gemini_client.models.generate_content(
                        model=_retrieval_model_id,
                        contents=captured,
                        config=config,
                    ),
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"_run_doc_sub_agent LLM call timed out (attempt {attempt})")
            break
        except Exception as e:
            logger.error(f"_run_doc_sub_agent LLM call failed (attempt {attempt}): {e}")
            break

        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.content is None:
            break

        # Extract reasoning text the LLM produced in this turn
        reasoning_text = " ".join(
            p.text for p in (candidate.content.parts or []) if hasattr(p, "text") and p.text
        ).strip()

        tool_calls = [p for p in (candidate.content.parts or []) if p.function_call]
        if not tool_calls:
            return {
                "answer": reasoning_text or "No relevant document found for the current request.",
                "doc_results": all_raw_docs,
                "attempts": attempts,
                "steps": steps,
            }

        contents.append(candidate.content)

        tool_response_parts = []
        for part in tool_calls:
            fc = part.function_call
            q = fc.args.get("query", user_query)
            k = int(fc.args.get("top_k", 3))
            text, raw = await loop.run_in_executor(None, lambda qq=q, kk=k: _sync_retrieve_docs(qq, kk))
            all_raw_docs.extend(raw)
            tool_response_parts.append(
                types.Part(function_response=types.FunctionResponse(name=fc.name, response={"result": text}))
            )
            step = {"round": attempt, "tool_query": q, "num_results": len(raw), "reasoning": reasoning_text}
            steps.append(step)
            if event_queue is not None:
                await event_queue.put({"type": "round", "query_idx": query_idx, "agent": "doc", **step})
        contents.append(types.Content(role="user", parts=tool_response_parts))

    return {
        "answer": "No relevant document found for the current request.",
        "doc_results": all_raw_docs,
        "attempts": attempts,
        "steps": steps,
    }


sdk = CopilotKitRemoteEndpoint(
    actions=[
        Action(
            name="search_faqs",
            description="Search the FAQ knowledge base for relevant question-answer pairs.",
            handler=search_faqs,
            parameters=[{
                "name": "query",
                "type": "string",
                "description": "The search query to find relevant FAQ entries.",
                "required": True,
            }],
        ),
        Action(
            name="search_documents",
            description="Search the document knowledge base for relevant passages.",
            handler=search_documents,
            parameters=[{
                "name": "query",
                "type": "string",
                "description": "The search query to find relevant document passages.",
                "required": True,
            }],
        ),
    ]
)

app = FastAPI(title="URASys CopilotKit Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
add_fastapi_endpoint(app, sdk, "/copilotkit_remote")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "faq_retriever": faq_retriever is not None,
        "doc_retriever": doc_retriever is not None,
        "sub_agent_llm": _gemini_client is not None,
        "retrieval_model": _retrieval_model_id,
    }


from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/search/faqs")
def rest_search_faqs(req: SearchRequest):
    if faq_retriever is None:
        return {"results": [], "error": "FAQ retriever not available."}
    try:
        results = faq_retriever.retrieve_faqs(req.query, top_k=req.top_k)
        items = []
        for faq in results.faqs:
            items.append({
                "question": faq.source_node.question,
                "answer": faq.source_node.answer,
                "score": round(float(faq.score), 4) if hasattr(faq, "score") else None,
            })
        return {"results": items, "total": len(items)}
    except Exception as e:
        logger.error(f"REST search_faqs error: {e}")
        return {"results": [], "error": str(e)}


@app.post("/search/documents")
def rest_search_documents(req: SearchRequest):
    if doc_retriever is None:
        return {"results": [], "error": "Document retriever not available."}
    try:
        results = doc_retriever.retrieve_documents(req.query, top_k=req.top_k)
        items = []
        for doc in results.documents:
            items.append({
                "chunk": doc.source_node.chunk,
                "score": round(float(doc.score), 4) if hasattr(doc, "score") else None,
            })
        return {"results": items, "total": len(items)}
    except Exception as e:
        logger.error(f"REST search_documents error: {e}")
        return {"results": [], "error": str(e)}


class ParallelSearchRequest(BaseModel):
    queries: List[str]
    top_k: int = 5


@app.post("/search/parallel")
async def rest_search_parallel(req: ParallelSearchRequest):
    """
    For each sub-query, run FAQ Sub-Agent and Document Sub-Agent concurrently.

    Each sub-agent implements the paper's iterative retrieval loop
    (Section 3.2 / Algorithm 1):
      - Prompted with paper system prompt (FAQ_SEARCH_INSTRUCTION_PROMPT /
        DOCUMENT_SEARCH_INSTRUCTION_PROMPT)
      - Issues tool calls to its retrieval function (up to max_retries times)
      - Reformulates the query if the previous results were unsatisfactory
      - Returns grounded text that becomes evidence for the Manager LLM

    Response per query:
      faq_answer  / doc_answer   — grounded text from each sub-agent (Manager sees these)
      faq_results / doc_results  — raw retrieved items (UI display only)
      faq_attempts/ doc_attempts — how many tool-call rounds were needed
    """
    async def search_one(query: str) -> dict:
        faq_task = asyncio.create_task(_run_faq_sub_agent(query))
        doc_task = asyncio.create_task(_run_doc_sub_agent(query))
        faq_out, doc_out = await asyncio.gather(faq_task, doc_task)
        return {
            "query": query,
            "faq_answer": faq_out["answer"],
            "doc_answer": doc_out["answer"],
            "faq_results": faq_out["faq_results"],
            "doc_results": doc_out["doc_results"],
            "faq_attempts": faq_out["attempts"],
            "doc_attempts": doc_out["attempts"],
            "faq_steps": faq_out.get("steps", []),
            "doc_steps": doc_out.get("steps", []),
        }

    if not req.queries:
        return {"query_results": [], "total_queries": 0}

    query_results = await asyncio.gather(*[search_one(q) for q in req.queries])
    return {"query_results": list(query_results), "total_queries": len(query_results)}


@app.post("/search/parallel/stream")
async def rest_search_parallel_stream(req: ParallelSearchRequest):
    """
    SSE streaming variant of /search/parallel.
    Emits one JSON event per line as each sub-agent round completes:
      {"type":"round",      "query_idx":N, "agent":"faq"|"doc", "round":R, "tool_query":"...", "num_results":K, "reasoning":"..."}
      {"type":"query_done", "query_idx":N, "query":"...", "faq_answer":"...", "doc_answer":"...", ...}
      {"type":"done",       "total_queries":N}
    """
    if not req.queries:
        async def _empty():
            yield f"data: {json.dumps({'type': 'done', 'total_queries': 0})}\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    event_queue: asyncio.Queue = asyncio.Queue()

    async def run_all():
        async def search_one(query_idx: int, query: str):
            faq_task = asyncio.create_task(
                _run_faq_sub_agent(query, event_queue=event_queue, query_idx=query_idx)
            )
            doc_task = asyncio.create_task(
                _run_doc_sub_agent(query, event_queue=event_queue, query_idx=query_idx)
            )
            faq_out, doc_out = await asyncio.gather(faq_task, doc_task)
            await event_queue.put({
                "type": "query_done",
                "query_idx": query_idx,
                "query": query,
                "faq_answer": faq_out["answer"],
                "doc_answer": doc_out["answer"],
                "faq_results": faq_out["faq_results"],
                "doc_results": doc_out["doc_results"],
                "faq_attempts": faq_out["attempts"],
                "doc_attempts": doc_out["attempts"],
                "faq_steps": faq_out.get("steps", []),
                "doc_steps": doc_out.get("steps", []),
            })

        await asyncio.gather(*[search_one(i, q) for i, q in enumerate(req.queries)])
        await event_queue.put(None)  # sentinel

    asyncio.create_task(run_all())

    async def event_generator():
        while True:
            event = await event_queue.get()
            if event is None:
                yield f"data: {json.dumps({'type': 'done', 'total_queries': len(req.queries)})}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main():
    uvicorn.run(app, host="0.0.0.0", port=8006, reload=False)


if __name__ == "__main__":
    main()

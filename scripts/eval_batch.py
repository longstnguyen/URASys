"""
URASys CLI — Interactive query testing + Batch evaluation with metrics.

Two main commands:

  interactive  — REPL where user types queries, sees live pipeline visualization
  eval         — Batch evaluation across retrieval strategies

────────────────────────────────────────────────────────────────────

Interactive mode:
    python scripts/eval_batch.py interactive
    python scripts/eval_batch.py i --strategy hybrid

Eval mode:
    # Compare all strategies on 50 samples (LLM-as-a-Judge, default)
    python scripts/eval_batch.py eval --dataset datasets/SQuAD2_1000.csv --n 50

    # Also compute EM/F1 (only recommended for normal answerable datasets)
    python scripts/eval_batch.py eval --dataset datasets/SQuAD2_1000.csv --n 50 --em-f1

    # Ambiguous dataset — auto-detected via 'info' column, LLM-as-a-Judge only
    python scripts/eval_batch.py eval --dataset datasets/SQuAD2_ambiguous.csv --n 50

    # Export results (always saved to --output, default: results/<dataset>_<strategy>_<timestamp>.json)
    python scripts/eval_batch.py eval --dataset datasets/SQuAD2_1000.csv --n 100 --output results.json
"""
import argparse
import asyncio
import json
import os
import re
import string
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import List, Optional

# Suppress noisy loguru/transformers warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if not os.environ.get("ENVIRONMENT_FILE"):
    os.environ["ENVIRONMENT_FILE"] = str(ROOT / "environments" / ".env")

# Suppress noisy library loggers
from loguru import logger as _loguru_logger
_loguru_logger.disable("urasys")

# ── ANSI helpers ───────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[31m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    BLUE   = "\033[34m"
    PURPLE = "\033[35m"
    CYAN   = "\033[36m"
    WHITE  = "\033[37m"
    BG_DK  = "\033[48;5;236m"


def _trunc(s: str, n: int = 80) -> str:
    return s[:n] + "…" if len(s) > n else s


def _bar(pct: float, w: int = 25) -> str:
    filled = int(pct * w)
    return f"{C.GREEN}{'█' * filled}{C.DIM}{'░' * (w - filled)}{C.RESET}"


# ── Globals ────────────────────────────────────────────────
_faq_retriever = None
_doc_retriever = None
_baseline_faq_retriever = None
_baseline_doc_retriever = None
_embedder = None
_faq_bm25 = None
_doc_bm25 = None
_baseline_faq_bm25 = None
_baseline_doc_bm25 = None
_vector_db = None
_gemini_client = None
_retrieval_model_id = "gemini-2.5-flash"
_SUB_AGENT_MAX_RETRIES = 3
_faq_col = None
_doc_col = None
_baseline_faq_col = None
_baseline_doc_col = None
_baseline_ready = False


def _init():
    global _faq_retriever, _doc_retriever, _embedder, _faq_bm25, _doc_bm25
    global _vector_db, _gemini_client, _retrieval_model_id, _faq_col, _doc_col
    global _baseline_doc_retriever
    global _baseline_doc_bm25
    global _baseline_doc_col, _baseline_ready

    from urasys.config.system_config import SETTINGS
    from urasys.config.utils import get_milvus_config
    from urasys.core.model_clients.bm25 import BM25Client
    from urasys.core.model_clients.embedder.openai import OpenAIClientConfig, OpenAIEmbedder
    from urasys.core.retriever.faq_retriever import FAQRetriever
    from urasys.core.retriever.document_retriever import DocumentRetriever
    from urasys.utils.base_class import ModelsConfig
    from urasys.utils.database_clients.milvus import MilvusVectorDatabase

    config_path = ROOT / "src/urasys/config/models_config.json"
    with open(config_path) as f:
        models_config = ModelsConfig.from_dict(json.load(f))
    embedding_cfg = models_config.embedding_config

    _embedder = OpenAIEmbedder(config=OpenAIClientConfig(
        api_key=SETTINGS.OPENAI_API_KEY, model=embedding_cfg.model_id,
    ))
    _vector_db = MilvusVectorDatabase(config=get_milvus_config(SETTINGS))

    faq_bm25_path = ROOT / "src/urasys/data/bm25/faq/state_dict.json"
    doc_bm25_path = ROOT / "src/urasys/data/bm25/document/state_dict.json"
    if not faq_bm25_path.exists() or not doc_bm25_path.exists():
        print(f"{C.RED}BM25 state dicts not found. Run: python scripts/index_offline.py{C.RESET}")
        sys.exit(1)
    _faq_bm25 = BM25Client(local_path=str(faq_bm25_path), init_without_load=False)
    _doc_bm25 = BM25Client(local_path=str(doc_bm25_path), init_without_load=False)

    _faq_col = SETTINGS.MILVUS_COLLECTION_FAQ_NAME
    _doc_col = SETTINGS.MILVUS_COLLECTION_DOCUMENT_NAME

    _faq_retriever = FAQRetriever(collection_name=_faq_col, embedder=_embedder, bm25_client=_faq_bm25, vector_db=_vector_db)
    _doc_retriever = DocumentRetriever(collection_name=_doc_col, embedder=_embedder, bm25_client=_doc_bm25, vector_db=_vector_db)

    # ── Baseline collection (document only — plain RAG) ──
    baseline_doc_bm25_path = ROOT / "src/urasys/data/bm25/baseline_document/state_dict.json"
    if baseline_doc_bm25_path.exists():
        _baseline_doc_bm25 = BM25Client(local_path=str(baseline_doc_bm25_path), init_without_load=False)
        _baseline_doc_col = "baseline_document_data"
        _baseline_doc_retriever = DocumentRetriever(collection_name=_baseline_doc_col, embedder=_embedder, bm25_client=_baseline_doc_bm25, vector_db=_vector_db)
        _baseline_ready = True
        print(f"  {C.GREEN}Baseline collection loaded.{C.RESET}")
    else:
        print(f"  {C.YELLOW}⚠ Baseline not indexed \u2014 run: python scripts/index_baseline.py{C.RESET}")

    from google import genai
    _gemini_client = genai.Client(api_key=SETTINGS.GEMINI_API_KEY)
    if hasattr(models_config, "llm_config") and "retrieval_llm" in models_config.llm_config:
        _retrieval_model_id = models_config.llm_config["retrieval_llm"].model_id


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RETRIEVAL STRATEGIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _search_bm25_only(query: str, top_k: int = 5) -> dict:
    """BM25 sparse search only — baseline (document only, no FAQ)."""
    if not _baseline_ready:
        return {"query": query, "results": [], "predicted_answer": "(baseline not indexed)"}
    from urasys.utils.database_clients.base_class import EmbeddingData, EmbeddingType
    from urasys.utils.database_clients.milvus.utils import MetricType
    doc_sparse = _baseline_doc_bm25.encode_queries([query])[0]
    doc_results = _vector_db.hybrid_search_vectors(
        embedding_data=[EmbeddingData(field_name="chunk_sparse_embedding", embeddings=doc_sparse, embedding_type=EmbeddingType.SPARSE)],
        output_fields=["chunk_id", "chunk"], top_k=top_k, collection_name=_baseline_doc_col, metric_type=MetricType.IP,
    )
    return _doc_results_to_output(query, doc_results, top_k)


def _search_dense_only(query: str, top_k: int = 5) -> dict:
    """Dense (OpenAI embedding) search only — baseline (document only, no FAQ)."""
    if not _baseline_ready:
        return {"query": query, "results": [], "predicted_answer": "(baseline not indexed)"}
    from urasys.utils.database_clients.base_class import EmbeddingData, EmbeddingType
    from urasys.utils.database_clients.milvus.utils import MetricType
    dense_emb = _embedder.get_query_embedding(query)
    doc_results = _vector_db.hybrid_search_vectors(
        embedding_data=[EmbeddingData(field_name="chunk_dense_embedding", embeddings=dense_emb, embedding_type=EmbeddingType.DENSE)],
        output_fields=["chunk_id", "chunk"], top_k=top_k, collection_name=_baseline_doc_col, metric_type=MetricType.IP,
    )
    return _doc_results_to_output(query, doc_results, top_k)


def _search_hybrid(query: str, top_k: int = 5) -> dict:
    """Hybrid BM25 + Dense with RRF — baseline (document only, no FAQ)."""
    if not _baseline_ready:
        return {"query": query, "results": [], "predicted_answer": "(baseline not indexed)"}
    doc_res = _baseline_doc_retriever.retrieve_documents(query, top_k=top_k)
    merged = []
    for d in doc_res.documents:
        merged.append({"text": d.source_node.chunk, "source": "document",
                       "detail": "", "score": round(float(d.score), 4)})
    merged.sort(key=lambda x: x["score"], reverse=True)
    merged = merged[:top_k]
    return {"query": query, "results": merged,
            "predicted_answer": merged[0]["text"] if merged else ""}


def _doc_results_to_output(query: str, doc_raw: list, top_k: int) -> dict:
    """Convert raw document results into output format for baselines."""
    merged = []
    for r in doc_raw:
        merged.append({"text": r.get("chunk", ""), "source": "document",
                       "detail": "", "score": round(float(r.get("_score", 0)), 4)})
    merged.sort(key=lambda x: x["score"], reverse=True)
    merged = merged[:top_k]
    return {"query": query, "results": merged,
            "predicted_answer": merged[0]["text"] if merged else ""}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BASELINE LLM GENERATION — standard RAG (retrieve → prompt → answer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_BASELINE_RAG_SYSTEM = """\
You are a helpful QA assistant. Answer the user's question using ONLY the provided context passages below.

Rules:
1. If the context contains a clear answer, provide it concisely.
2. If the context does NOT contain enough information to answer, reply exactly: "Không tìm thấy thông tin phù hợp trong cơ sở dữ liệu."
3. If the question is too vague or ambiguous to answer (missing subject, time frame, or scope), ask a brief clarifying question to help the user specify their intent.
4. Do NOT fabricate information. Only use what is in the context.
5. Respond in the same language as the user's question.
"""


def _baseline_generate(query: str, results: list, history: list = None) -> str:
    """Standard RAG: feed retrieved chunks to LLM and generate an answer."""
    if not results:
        return "Không tìm thấy thông tin phù hợp trong cơ sở dữ liệu."

    from google.genai import types

    # Build context from retrieved chunks
    context_parts = []
    for i, r in enumerate(results):
        text = r.get("text", "")
        if text:
            context_parts.append(f"[Passage {i+1}]\n{text}")
    context_str = "\n\n".join(context_parts)

    messages: list[types.Content] = []

    # Prior conversation turns (if interactive mode)
    if history:
        for h in history[-3:]:
            messages.append(types.Content(role="user", parts=[types.Part(text=h["query"])]))
            messages.append(types.Content(role="model", parts=[types.Part(text=h["answer"])]))

    # Current user question with context
    user_msg = f"Context:\n{context_str}\n\nQuestion: {query}"
    messages.append(types.Content(role="user", parts=[types.Part(text=user_msg)]))

    try:
        response = _gemini_client.models.generate_content(
            model=_retrieval_model_id,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=_BASELINE_RAG_SYSTEM,
                temperature=0.2,
                top_p=0.1,
            ),
        )
        text = response.text.strip() if response.text else ""
        return text or "Không tìm thấy thông tin phù hợp trong cơ sở dữ liệu."
    except Exception:
        return "Không tìm thấy thông tin phù hợp trong cơ sở dữ liệu."


async def _search_urasys(query: str, top_k: int = 5) -> dict:
    """Full URASys agentic pipeline with sub-agent LLM loops."""
    from google.genai import types
    from urasys.prompts.query import FAQ_SEARCH_INSTRUCTION_PROMPT, DOCUMENT_SEARCH_INSTRUCTION_PROMPT
    loop = asyncio.get_running_loop()

    async def _run_agent(user_q: str, agent_type: str) -> dict:
        is_faq = agent_type == "faq"
        prompt_tpl = FAQ_SEARCH_INSTRUCTION_PROMPT if is_faq else DOCUMENT_SEARCH_INSTRUCTION_PROMPT
        system_prompt = prompt_tpl.format(max_retries=_SUB_AGENT_MAX_RETRIES)
        tool_name = "faq_retrieval_tool" if is_faq else "document_retrieval_tool"
        tool_desc = "Retrieve top K relevant FAQs" if is_faq else "Retrieve top K relevant document passages"
        contents = [types.Content(role="user", parts=[types.Part(text=user_q)])]
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[types.Tool(function_declarations=[types.FunctionDeclaration(
                name=tool_name, description=tool_desc,
                parameters=types.Schema(type=types.Type.OBJECT, properties={
                    "query": types.Schema(type=types.Type.STRING, description="Search query."),
                    "top_k": types.Schema(type=types.Type.INTEGER, description="Number of results."),
                }, required=["query"]),
            )])], temperature=0.1,
        )
        all_raw, steps, attempts = [], [], 0
        for attempt_num in range(1, _SUB_AGENT_MAX_RETRIES + 1):
            attempts = attempt_num
            try:
                captured = list(contents)
                response = await asyncio.wait_for(loop.run_in_executor(None,
                    lambda: _gemini_client.models.generate_content(model=_retrieval_model_id, contents=captured, config=config)),
                    timeout=30.0)
            except Exception:
                break
            candidate = response.candidates[0] if response.candidates else None
            if not candidate or not candidate.content:
                break
            reasoning = " ".join(p.text for p in (candidate.content.parts or []) if hasattr(p, "text") and p.text).strip()
            tool_calls = [p for p in (candidate.content.parts or []) if p.function_call]
            if not tool_calls:
                return {"answer": reasoning or f"No relevant {'FAQ' if is_faq else 'document'} found.", "results": all_raw, "attempts": attempts, "steps": steps}
            contents.append(candidate.content)
            tool_resp_parts = []
            for part in tool_calls:
                fc = part.function_call
                q = fc.args.get("query", user_q)
                k = int(fc.args.get("top_k", top_k))
                if is_faq:
                    res = _faq_retriever.retrieve_faqs(q, top_k=k)
                    raw = [{"question": f.source_node.question, "answer": f.source_node.answer, "score": round(float(f.score), 4)} for f in res.faqs]
                    text = "\n\n".join(f"[FAQ {i+1}] Q: {r['question']}\nA: {r['answer']}" for i, r in enumerate(raw)) or "No FAQs found."
                else:
                    res = _doc_retriever.retrieve_documents(q, top_k=k)
                    raw = [{"chunk": d.source_node.chunk, "score": round(float(d.score), 4)} for d in res.documents]
                    text = "\n\n".join(f"[Passage {i+1}]\n{r['chunk']}" for i, r in enumerate(raw)) or "No documents found."
                all_raw.extend(raw)
                tool_resp_parts.append(types.Part(function_response=types.FunctionResponse(name=fc.name, response={"result": text})))
                steps.append({"round": attempt_num, "query": q, "results": len(raw), "reasoning": reasoning[:80]})
            contents.append(types.Content(role="user", parts=tool_resp_parts))
        return {"answer": f"No relevant {'FAQ' if is_faq else 'document'} found.", "results": all_raw, "attempts": attempts, "steps": steps}

    faq_out, doc_out = await asyncio.gather(_run_agent(query, "faq"), _run_agent(query, "doc"))
    return {
        "query": query, "faqs": faq_out["results"], "docs": doc_out["results"],
        "faq_answer": faq_out["answer"], "doc_answer": doc_out["answer"],
        "faq_attempts": faq_out["attempts"], "doc_attempts": doc_out["attempts"],
        "faq_steps": faq_out["steps"], "doc_steps": doc_out["steps"],
    }



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MANAGER AGENT SYSTEM PROMPT — identical to frontend/app/page.tsx
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MANAGER_SYSTEM = """\
# Persona
You are the "AI Assistant," an expert AI focused on efficiently and accurately answering questions using the provided context passages.

# Current State
- You have exactly ONE search call per question. The sub-agents handle up to 3 internal reformulation rounds each.

# The Supreme Goal: The "Just Enough" Principle
Your absolute highest priority is to answer the user's *specific, underlying need*, not just the broad words they use. You must act as a **guide**, not an information dump. This means:
- If a query is broad, your job is to **help the user specify it.**
- If a query is specific, your job is to **answer it directly.**
- **NEVER** dump a summary of all found information and then ask "what do you want to know more about?". This is a critical failure.

# Core Directives
1.  **Search is for Understanding:** Your first search on a broad topic is not to find an answer, but to **discover the available categories/options** to guide the user.
2.  **Troubleshoot Vague Failures:** If a search fails because the user's query is incomplete, ask for more clues.
3.  **Evidence-Based Actions:** All answers and examples MUST come from the retrieved context passages.
4.  **Language and Persona Integrity:**
    *   All responses **MUST** be in **language based on an user**.
    *   **Self-reference:** Use the pronoun **"I"** to refer to yourself. Only state your full name if asked directly.
    *   **Expert Tone and Phrasing:** You **MUST** speak from a position of knowledge, as a representative of the university.
        *   **DO:** Use confident, knowledgeable phrasing like: *"Now, I...", "About [topic], I see that..."*
        *   **AVOID:** **NEVER** use phrases that imply real-time discovery. **FORBIDDEN** phrases include: *"I search...", "I have...", "In my researching,..."*
    *   **Conceal Internal Mechanics:** **NEVER** mention your tools or processes.
5.  **Queries:** All search queries **MUST** be in language based on user.
6.  **No Fabrication:** If you cannot find information, state it clearly.

# Decision-Making Workflow: A Strict Gate System

**Step 0: Check for Meta-Questions (FIRST)**
*   If the user's message is about **YOU** (your capabilities, your topics, what you know, what you can help with) — do NOT call search_information. Answer directly and briefly, then invite them to ask a specific question.

**Step 1: Analyze Request & Search**
*   For all other questions: Call search_information with attempt: 1 to understand the information landscape.
*   **DECOMPOSITION IS CRITICAL:** If the question compares, contrasts, or relates two or more entities (people, places, dates, concepts), you MUST decompose into separate keyword phrases — one per entity. Each sub-query should be a SHORT keyword phrase (2-5 words), NOT a full question. For example: "Were X and Y of the same nationality?" → query1='X nationality', query2='Y nationality'.

**Step 2: Evaluate Results & Choose a Path (Choose ONLY ONE)**
The tool returns per-query results with these fields:
- faq_answer / doc_answer — grounded answers from the sub-agents. If "No relevant document found", the sub-agent could not match evidence to the query.
- num_faq_results / num_doc_results — how many items the retriever found (even if the sub-agent deemed them irrelevant).

**Key signal:** If both answers say "No relevant..." BUT the counts are > 0, it means the knowledge base HAS data but the query was too vague/broad to match anything specific. This is a strong signal for PATH B or PATH C (ask clarification), NOT PATH D.

*   **PATH A: The "Specific Answer" Gate** — IsSpecific(q) AND HasDirectAnswer(E) AND Consistent(E)
    *   **CONDITION:** The user's query points to **exactly one topic/item** (specific) AND you found a direct answer AND the FAQ and Document evidence are **consistent** (agree or complement each other — no contradictions between sources).
    *   **GUARD:** A query that is just 1-2 generic words (e.g., "lịch sử", "chiến tranh", "khoa học") is NEVER considered specific, even if the sub-agents returned answers. Such queries MUST go to PATH B or C.
    *   **ACTION:** Synthesize the answer from the evidence. Your turn ends.

*   **PATH B: The "Clarification" Gate** — IsBroad(q) AND RevealCategories(E)
    *   **CONDITION:** The user's query covers a **topic with multiple distinct sub-types/categories** — i.e., the same question could have several different specific answers depending on which sub-type the user means (e.g., "what fees are there?" when the KB has tuition, lab, and registration fees) AND the search revealed those distinct categories.
    *   **PRIORITY: PATH B takes precedence over PATH A when multiple categories exist.** Even if you found some information, if the query is genuinely broad and multiple distinct sub-topics appear, ask for clarification instead of picking one arbitrarily.
    *   **ACTION:**
        1.  **STOP.**
        2.  Ask a clarifying question using an **Expert Tone** — list ONLY the **NAMES** of the categories/sub-topics you found.
        3.  **STRICTLY FORBIDDEN:** Do not include specific values, numbers, or details in the clarification question.

*   **PATH C: The "Clarify Vague" Gate** — IsVague(q) AND (Insufficient(E) OR counts > 0)
    *   **CONDITION:** The user's query is **vague, incomplete, or missing context** (e.g., "phát triển như thế nào?" — develop how? what topic?). The query lacks a clear subject, time frame, or scope, BUT the topic itself is plausibly within the knowledge base's domain.
    *   **STRONG SIGNAL:** If num_faq_results or num_doc_results > 0 AND both answers say "No relevant...", BUT the retrieved items' topics are related to the user's query — the knowledge base HAS relevant content, the query just needs to be more specific. Ask for clarification.
    *   **ACTION:** Ask a specific clarifying question. For example: "Bạn muốn hỏi về sự phát triển của lĩnh vực nào? Ví dụ: ngành Khoa học Máy tính, hoạt động nghiên cứu, hay cơ sở vật chất?" Frame it as an expert guiding the user.

*   **PATH D: The "No Information" Gate**
    *   **CONDITION:** One of:
        (a) Both answers say "No relevant..." AND num_faq_results = 0 AND num_doc_results = 0 (the knowledge base truly has nothing).
        (b) The query is **clearly off-topic** — the subject (e.g., cooking recipes, sports scores, entertainment) has no plausible connection to the knowledge base's domain. Even if counts > 0, those results are just retriever noise, not real matches.
    *   **ACTION:** Politely inform the user you could not find the information in the knowledge base.
"""


def _synthesize_final_answer(query: str, faq_answer: str, doc_answer: str,
                             history: list = None,
                             num_faq: int = 0, num_doc: int = 0) -> str:
    """Manager Agent synthesis — uses the SAME system prompt as the UI (page.tsx).

    The evidence is formatted identically to the search_information tool result
    so the Manager LLM receives the same context as in the frontend.
    Eval runs identically to the UI: the model may answer, clarify, or refuse.
    LLM-as-a-Judge then evaluates the response appropriately.
    """
    from google.genai import types

    system = _MANAGER_SYSTEM

    # ── Build the conversation as the Manager Agent would see it ──
    messages: list[types.Content] = []

    # Prior conversation turns (if interactive mode)
    if history:
        for h in history[-3:]:
            messages.append(types.Content(role="user", parts=[types.Part(text=h["query"])]))
            messages.append(types.Content(role="model", parts=[types.Part(text=h["answer"])]))

    # Current user question
    messages.append(types.Content(role="user", parts=[types.Part(text=query)]))

    # Simulate the tool call + tool result as the Manager would see them
    # (model made a search_information call → got this result)
    import json as _json
    tool_result = {
        "total_queries": 1,
        "query_results": [{
            "query": query,
            "faq_answer": faq_answer or "No relevant FAQ found for the current request.",
            "doc_answer": doc_answer or "No relevant document found for the current request.",
            "num_faq_results": num_faq,
            "num_doc_results": num_doc,
        }],
    }
    # Model turn: tool call
    messages.append(types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(
            name="search_information",
            args={"query1": query, "attempt": 1},
        ))
    ]))
    # Tool response
    messages.append(types.Content(role="user", parts=[
        types.Part(function_response=types.FunctionResponse(
            name="search_information",
            response=tool_result,
        ))
    ]))

    try:
        response = _gemini_client.models.generate_content(
            model=_retrieval_model_id,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.2,
                top_p=0.1,
            ),
        )
        text = response.text.strip() if response.text else ""
        if not text:
            faq_valid = faq_answer and "No relevant" not in faq_answer
            return faq_answer if faq_valid else (doc_answer or "Không tìm thấy thông tin phù hợp.")
        return text
    except Exception:
        faq_valid = faq_answer and "No relevant" not in faq_answer
        return faq_answer if faq_valid else (doc_answer or "Không tìm thấy thông tin phù hợp.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METRICS — EM + F1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"\b(a|an|the|là|của|và|các|những|này|đó|được)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def compute_em(prediction: str, ground_truth: str) -> float:
    return float(_normalize(prediction) == _normalize(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gt_tokens = _normalize(ground_truth).split()
    if not gt_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _best_answer(result: dict) -> str:
    """Extract best answer from results for EM/F1 comparison."""
    # URASys: prefer synthesized final answer
    if result.get("final_answer") and result["final_answer"]:
        return result["final_answer"]
    # Baselines: unified "results" list with top answer already set
    if "predicted_answer" in result and result["predicted_answer"]:
        return result["predicted_answer"]
    # URASys fallback: prefer grounded LLM answer
    if result.get("faq_answer") and "No relevant" not in result["faq_answer"]:
        return result["faq_answer"]
    if result.get("doc_answer") and "No relevant" not in result["doc_answer"]:
        return result["doc_answer"]
    return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM-AS-A-JUDGE  (Paper §4.2 + Appendix prompts)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Standard prompt — for normal QA and unanswerable (paper Figure 5)
_JUDGE_STANDARD_SYSTEM = """### Role
You are an expert language model evaluator. Your task is to evaluate the model prediction given a ground truth.
### Instruction
- If the prediction is "no answer" -> False
- If the prediction contains the ground truth verbatim -> True
- If the prediction paraphrases the ground truth -> True
- If the prediction contradicts the ground truth -> False
- Evaluation is language-agnostic: ignore whether the prediction is in English or Vietnamese.
### Note
- The prediction may contain extra explanatory text: ignore it.
"""

# Ambiguous prompt — for datasets with 'info' column (paper Figure 6)
_JUDGE_AMBIGUOUS_SYSTEM = """You are an evaluator for an ambiguous question answering system.
Evaluation Rules:
1. You are given: the question, a list of required info items, the correct answer, and the model's prediction.
2. If the prediction is a clarification question:
    - CORRECT if it explicitly asks for at least one required info item.
    - INCORRECT if it does not ask for any required info items.
3. If the prediction is a direct answer:
    - CORRECT if it matches or clearly paraphrases the correct answer.
    - INCORRECT otherwise.
4. If the prediction refuses to answer, says "I don't know", or indicates insufficient information:
    - Always INCORRECT.
5. Ignore the language used; evaluate purely based on content.
6. Output exactly one word: CORRECT or INCORRECT.
"""

_judge_client = None   # OpenAI client for GPT-4o judge
_JUDGE_MODEL = "gpt-4o"


def _init_judge():
    """Initialize GPT-4o judge client (paper uses GPT-4o, temp=0)."""
    global _judge_client
    try:
        from openai import OpenAI
        from urasys.config.system_config import SETTINGS
        _judge_client = OpenAI(api_key=SETTINGS.OPENAI_API_KEY)
    except Exception as e:
        print(f"  {C.YELLOW}⚠ LLM-as-a-Judge unavailable: {e}{C.RESET}")


def judge_standard(question: str, ground_truth: str, prediction: str) -> bool:
    """Standard LLM-as-a-Judge (paper Figure 5): True/False."""
    if _judge_client is None:
        return False
    user_msg = (
        f"Question: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Prediction: {prediction}"
    )
    try:
        resp = _judge_client.chat.completions.create(
            model=_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_STANDARD_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0, max_tokens=32,
        )
        answer = resp.choices[0].message.content.strip().lower()
        return "true" in answer
    except Exception:
        return False


def judge_ambiguous(question: str, info: str, ground_truth: str, prediction: str) -> bool:
    """Ambiguous LLM-as-a-Judge (paper Figure 6): CORRECT/INCORRECT."""
    if _judge_client is None:
        return False
    user_msg = (
        f"Question: {question}\n"
        f"Required Info Items: {info}\n"
        f"Correct Answer: {ground_truth}\n"
        f"Model Prediction: {prediction}"
    )
    try:
        resp = _judge_client.chat.completions.create(
            model=_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_AMBIGUOUS_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0, max_tokens=32,
        )
        answer = resp.choices[0].message.content.strip().lower()
        return "correct" in answer and "incorrect" not in answer
    except Exception:
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTERACTIVE MODE — beautiful terminal REPL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_ICONS = {"pending": f"{C.DIM}○{C.RESET}", "running": f"{C.YELLOW}◐{C.RESET}",
          "done": f"{C.GREEN}●{C.RESET}", "error": f"{C.RED}✗{C.RESET}"}


def _step_line(state: str, label: str, detail: str = "", elapsed: float = 0) -> str:
    icon = _ICONS[state]
    t = f" {C.DIM}{elapsed:.1f}s{C.RESET}" if elapsed > 0 else ""
    d = f"  {C.DIM}{detail}{C.RESET}" if detail else ""
    return f"  {icon} {label}{t}{d}"


def _interactive_search(query: str, strategy: str = "urasys", top_k: int = 5, history: list = None):
    """Run a single query with live terminal visualization."""
    is_baseline = strategy in ("bm25", "dense", "hybrid")

    if is_baseline:
        steps = [
            {"label": "Encode Query",    "state": "pending", "detail": "", "t": 0.0},
            {"label": "Retrieve",        "state": "pending", "detail": "", "t": 0.0},
            {"label": "Generate Answer", "state": "pending", "detail": "", "t": 0.0},
        ]
    else:
        steps = [
            {"label": "Analyze Query",         "state": "pending", "detail": "", "t": 0.0},
            {"label": "FAQ Search Agent",      "state": "pending", "detail": "", "t": 0.0},
            {"label": "Document Search Agent", "state": "pending", "detail": "", "t": 0.0},
            {"label": "Synthesize Answer",     "state": "pending", "detail": "", "t": 0.0},
        ]
    n_lines = [0]

    def render():
        if n_lines[0] > 0:
            sys.stdout.write(f"\033[{n_lines[0]}A")
        lines = [f"  {C.BG_DK}{C.BOLD} Pipeline — {strategy.upper()} {C.RESET}", ""]
        for s in steps:
            lines.append(_step_line(s["state"], s["label"], s["detail"], s["t"]))
        lines.append("")
        for i in range(n_lines[0]):
            sys.stdout.write(f"\033[2K\n")
        if n_lines[0] > 0:
            sys.stdout.write(f"\033[{n_lines[0]}A")
        sys.stdout.write("\n".join(lines))
        sys.stdout.flush()
        n_lines[0] = len(lines)

    render()
    t0 = time.time()

    if is_baseline:
        # Step 1: Encode
        steps[0]["state"] = "running"
        steps[0]["detail"] = f'"{_trunc(query, 50)}"'
        render()
        t1 = time.time()
        if strategy == "bm25":
            result = _search_bm25_only(query, top_k)
        elif strategy == "dense":
            result = _search_dense_only(query, top_k)
        else:
            result = _search_hybrid(query, top_k)
        t_s = time.time() - t1
        steps[0]["state"] = "done"
        steps[0]["t"] = t_s
        steps[0]["detail"] = strategy
        # Step 2: Retrieve done
        steps[1]["state"] = "done"
        steps[1]["detail"] = f"{len(result['results'])} results"
        steps[1]["t"] = t_s
        render()
        # Step 3: LLM Generate
        steps[2]["state"] = "running"
        steps[2]["detail"] = "generating from context…"
        render()
        t2 = time.time()
        result["predicted_answer"] = _baseline_generate(query, result.get("results", []), history=history)
        steps[2]["state"] = "done"
        steps[2]["detail"] = "done"
        steps[2]["t"] = time.time() - t2
        render()

    else:
        # URASys agentic pipeline
        steps[0]["state"] = "running"
        steps[0]["detail"] = f'"{_trunc(query, 50)}"'
        render()
        steps[0]["state"] = "done"
        steps[0]["t"] = time.time() - t0
        steps[0]["detail"] = "agentic pipeline"
        steps[1]["state"] = "running"
        steps[2]["state"] = "running"
        render()
        t1 = time.time()
        result = asyncio.run(_search_urasys(query, top_k))
        t_s = time.time() - t1
        steps[1] = {"label": "FAQ Search Agent", "state": "done",
                     "detail": f"{len(result['faqs'])} FAQs, {result.get('faq_attempts', 1)} round(s)", "t": t_s}
        steps[2] = {"label": "Document Search Agent", "state": "done",
                     "detail": f"{len(result['docs'])} docs, {result.get('doc_attempts', 1)} round(s)", "t": t_s}
        render()
        # Step 4: Synthesize final answer
        steps[3]["state"] = "running"
        steps[3]["detail"] = "combining agents…"
        render()
        t2 = time.time()
        final = _synthesize_final_answer(
            query, result.get("faq_answer", ""), result.get("doc_answer", ""),
            history=history,
            num_faq=len(result.get("faqs", [])), num_doc=len(result.get("docs", [])))
        result["final_answer"] = final
        result["predicted_answer"] = final
        steps[3] = {"label": "Synthesize Answer", "state": "done", "detail": "done", "t": time.time() - t2}
        render()

    total = time.time() - t0
    print(f"\n\n  {C.BG_DK}{C.BOLD} Results — {total:.2f}s total {C.RESET}\n")

    if is_baseline:
        # Show retrieved passages
        if result["results"]:
            print(f"  {C.BLUE}{C.BOLD}Retrieved Passages{C.RESET}")
            for i, r in enumerate(result["results"][:5]):
                sc = f"{C.DIM}[{r['score']:.4f}]{C.RESET}"
                print(f"    {C.BOLD}{i+1}.{C.RESET} {sc}  {_trunc(r['text'], 75)}")
        else:
            print(f"    {C.RED}(no results){C.RESET}")
        # Show LLM-generated answer
        print(f"\n  {C.BG_DK}{C.BOLD} Final Answer {C.RESET}\n")
        print(f"  {C.GREEN}{result.get('predicted_answer', '')}{C.RESET}")
    else:
        # URASys: show FAQ agent + Document agent details
        print(f"  {C.PURPLE}{C.BOLD}FAQ Search Agent{C.RESET}  {C.DIM}({result.get('faq_attempts', '?')} rounds){C.RESET}")
        for s in result.get("faq_steps", []):
            print(f"    {C.DIM}R{s['round']}: \"{s['query']}\" → {s['results']} hits{C.RESET}")
        if result["faqs"]:
            for i, f in enumerate(result["faqs"][:3]):
                sc = f"{C.DIM}[{f['score']:.4f}]{C.RESET}" if f.get("score") else ""
                print(f"    {C.YELLOW}Q{i+1}:{C.RESET} {_trunc(f['question'], 70)}  {sc}")
                print(f"    {C.GREEN}A:{C.RESET}  {_trunc(f['answer'], 70)}")
        else:
            print(f"    {C.RED}(no results){C.RESET}")
        if result.get("faq_answer"):
            clr = C.GREEN if "No relevant" not in result["faq_answer"] else C.RED
            print(f"    {C.BOLD}→ Grounded:{C.RESET} {clr}{_trunc(result['faq_answer'], 100)}{C.RESET}")

        print()
        print(f"  {C.BLUE}{C.BOLD}Document Search Agent{C.RESET}  {C.DIM}({result.get('doc_attempts', '?')} rounds){C.RESET}")
        for s in result.get("doc_steps", []):
            print(f"    {C.DIM}R{s['round']}: \"{s['query']}\" → {s['results']} hits{C.RESET}")
        if result["docs"]:
            for i, d in enumerate(result["docs"][:3]):
                sc = f"{C.DIM}[{d['score']:.4f}]{C.RESET}" if d.get("score") else ""
                print(f"    {C.CYAN}P{i+1}:{C.RESET} {_trunc(d['chunk'], 80)}  {sc}")
        else:
            print(f"    {C.RED}(no results){C.RESET}")
        if result.get("doc_answer"):
            clr = C.GREEN if "No relevant" not in result["doc_answer"] else C.RED
            print(f"    {C.BOLD}→ Grounded:{C.RESET} {clr}{_trunc(result['doc_answer'], 100)}{C.RESET}")

        # ── Final synthesized answer ────────────────────
        print(f"\n  {C.BG_DK}{C.BOLD} Final Answer {C.RESET}\n")
        print(f"  {C.GREEN}{result.get('final_answer', '')}{C.RESET}")
    print()
    return result


def _run_interactive(strategy: str, top_k: int):
    print(f"\n{C.BG_DK}{C.BOLD}  URASys Interactive — type a query, press Enter  {C.RESET}")
    print(f"  {C.DIM}Strategy: {strategy} | Top-K: {top_k} | 'quit' to exit | ':s <...>' to switch | ':clear' to reset history{C.RESET}\n")
    cur = strategy
    history = []
    while True:
        try:
            query = input(f"  {C.PURPLE}❯{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}Bye!{C.RESET}")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        if query == ":clear":
            history.clear()
            print(f"  {C.GREEN}History cleared.{C.RESET}\n")
            continue
        if query.startswith(":s "):
            s = query[3:].strip().lower()
            if s in ("bm25", "dense", "hybrid", "urasys"):
                cur = s
                history.clear()
                print(f"  {C.GREEN}Switched to: {cur}{C.RESET}\n")
            else:
                print(f"  {C.RED}Use: bm25, dense, hybrid, urasys{C.RESET}\n")
            continue
        print()
        result = _interactive_search(query, cur, top_k, history=history)
        # Track conversation history for multi-turn
        ans = result.get("final_answer") or result.get("predicted_answer", "")
        if ans:
            history.append({"query": query, "answer": ans})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVAL MODE — batch evaluation with EM/F1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGIES = ["bm25", "dense", "hybrid", "urasys"]


def _run_one(query: str, strategy: str, top_k: int) -> dict:
    t0 = time.time()
    if strategy == "bm25":
        res = _search_bm25_only(query, top_k)
    elif strategy == "dense":
        res = _search_dense_only(query, top_k)
    elif strategy == "hybrid":
        res = _search_hybrid(query, top_k)
    elif strategy == "urasys":
        res = asyncio.run(_search_urasys(query, top_k))
        res["final_answer"] = _synthesize_final_answer(
            query, res.get("faq_answer", ""), res.get("doc_answer", ""),
            num_faq=len(res.get("faqs", [])), num_doc=len(res.get("docs", [])))
        res["predicted_answer"] = res["final_answer"] or _best_answer(res)
    else:
        res = {"query": query, "results": [], "predicted_answer": ""}
    # Baseline strategies: generate answer via LLM (standard RAG)
    if strategy in ("bm25", "dense", "hybrid"):
        res["predicted_answer"] = _baseline_generate(query, res.get("results", []))
    res["elapsed"] = round(time.time() - t0, 2)
    return res


def _run_eval(strategies: List[str], questions: List[str],
              ground_truths: Optional[List[str]], top_k: int, output_path: Optional[str],
              use_judge: bool = True, is_ambiguous: bool = False,
              info_items: Optional[List[str]] = None,
              use_em_f1: bool = False):
    """
    Batch evaluation.
    
    Default metric: LLM-as-a-Judge (paper §4.2).
    Optional: EM/F1 via --em-f1 (only meaningful for normal answerable questions;
              NOT recommended for ambiguous or unanswerable subsets).
    """
    has_gt = ground_truths is not None and len(ground_truths) == len(questions)
    all_results = {}

    # Detect unanswerable entries (answer = NaN, empty, or "No answer")
    unanswerable_mask = []
    if has_gt:
        for gt in ground_truths:
            is_unans = (gt is None or
                        (isinstance(gt, float) and str(gt) == 'nan') or
                        str(gt).strip().lower() in ('', 'nan', 'no answer'))
            unanswerable_mask.append(is_unans)
    else:
        unanswerable_mask = [False] * len(questions)

    for strat in strategies:
        is_baseline = strat != "urasys"
        print(f"\n{C.BG_DK}{C.BOLD}  Strategy: {strat.upper()}  {C.RESET}")
        mode_tag = "ambiguous" if is_ambiguous else "standard"
        em_tag = "on (answerable only)" if use_em_f1 and not is_ambiguous else "off"
        print(f"  {C.DIM}Mode: {mode_tag} | Judge: {'on' if use_judge else 'off'} | "
              f"EM/F1: {em_tag} | Unanswerable: {sum(unanswerable_mask)}/{len(questions)}{C.RESET}\n")
        results = []
        for i, q in enumerate(questions):
            pct = (i + 1) / len(questions)
            gt_str = str(ground_truths[i]) if has_gt and ground_truths[i] is not None else ""
            gt_hint = f"  {C.DIM}GT: {_trunc(gt_str, 30)}{C.RESET}" if has_gt else ""
            unans_tag = f" {C.YELLOW}[unans]{C.RESET}" if unanswerable_mask[i] else ""
            print(f"\r  {_bar(pct)} {i+1}/{len(questions)}{unans_tag}  "
                  f"{C.DIM}{_trunc(q, 40)}{C.RESET}{gt_hint}", end="", flush=True)
            r = _run_one(q, strat, top_k)
            pred = r.get("predicted_answer", "")

            if has_gt:
                r["ground_truth"] = gt_str
                r["is_unanswerable"] = unanswerable_mask[i]

                # ── EM / F1 (opt-in, only for normal answerable questions) ──
                if use_em_f1 and not is_ambiguous and not unanswerable_mask[i] and gt_str:
                    r["em"] = compute_em(pred, gt_str)
                    r["f1"] = compute_f1(pred, gt_str)

                # ── LLM-as-a-Judge ──
                if use_judge and _judge_client is not None:
                    if is_ambiguous:
                        info_str = info_items[i] if info_items else ""
                        r["judge"] = judge_ambiguous(q, info_str, gt_str, pred)
                    else:
                        r["judge"] = judge_standard(q, gt_str, pred)

            results.append(r)
        print()
        all_results[strat] = results

        # ── Per-strategy summary ──
        avg_t = sum(r["elapsed"] for r in results) / len(results)
        print(f"  {C.DIM}Avg time:{C.RESET} {avg_t:.2f}s/q")

        if has_gt:
            # Split into answerable / unanswerable
            ans_res = [r for r, u in zip(results, unanswerable_mask) if not u]
            unans_res = [r for r, u in zip(results, unanswerable_mask) if u]

            if use_em_f1 and not is_ambiguous and ans_res and any("em" in r for r in ans_res):
                em_res = [r for r in ans_res if "em" in r]
                avg_em = sum(r["em"] for r in em_res) / len(em_res)
                avg_f1 = sum(r["f1"] for r in em_res) / len(em_res)
                print(f"  {C.GREEN}EM:{C.RESET} {avg_em:.4f}  {C.GREEN}F1:{C.RESET} {avg_f1:.4f}  "
                      f"{C.DIM}(answerable: {len(em_res)}){C.RESET}")

            if use_judge and any("judge" in r for r in results):
                # Overall judge accuracy
                judged = [r for r in results if "judge" in r]
                acc_all = sum(r["judge"] for r in judged) / len(judged) if judged else 0
                print(f"  {C.CYAN}Judge (overall):{C.RESET} {acc_all:.4f}  ({sum(r['judge'] for r in judged)}/{len(judged)})")

                # Answerable subset
                judged_ans = [r for r, u in zip(results, unanswerable_mask) if not u and "judge" in r]
                if judged_ans:
                    acc_ans = sum(r["judge"] for r in judged_ans) / len(judged_ans)
                    print(f"  {C.CYAN}Judge (answerable):{C.RESET} {acc_ans:.4f}  ({sum(r['judge'] for r in judged_ans)}/{len(judged_ans)})")

                # Unanswerable subset
                judged_unans = [r for r, u in zip(results, unanswerable_mask) if u and "judge" in r]
                if judged_unans:
                    acc_unans = sum(r["judge"] for r in judged_unans) / len(judged_unans)
                    print(f"  {C.CYAN}Judge (unanswerable):{C.RESET} {acc_unans:.4f}  ({sum(r['judge'] for r in judged_unans)}/{len(judged_unans)})")

    # ── Comparison table ──
    if len(strategies) > 1:
        print(f"\n{C.BG_DK}{C.BOLD}  Comparison Table  {C.RESET}\n")
        has_em = use_em_f1 and not is_ambiguous and has_gt
        has_judge_col = use_judge and any("judge" in r for s in all_results.values() for r in s)
        hdr = f"  {'Strategy':<10}{'Time':>8}"
        if has_em:
            hdr += f"{'EM':>8}{'F1':>8}"
        if has_judge_col:
            hdr += f"{'Judge':>8}{'Unans':>8}{'Ambig':>8}" if not is_ambiguous else f"{'Judge':>8}"
        print(f"{C.BOLD}{hdr}{C.RESET}")
        col_w = 10 + 8 + (16 if has_em else 0) + (24 if has_judge_col and not is_ambiguous else 8 if has_judge_col else 0)
        print(f"  {'─' * col_w}")
        for strat in strategies:
            res = all_results[strat]
            at = sum(r["elapsed"] for r in res) / len(res)
            row = f"  {strat:<10}{at:>7.2f}s"
            if has_em:
                ans_r = [r for r, u in zip(res, unanswerable_mask) if not u]
                em = sum(r.get("em", 0) for r in ans_r) / len(ans_r) if ans_r else 0
                f1 = sum(r.get("f1", 0) for r in ans_r) / len(ans_r) if ans_r else 0
                row += f"{em:>8.4f}{f1:>8.4f}"
            if has_judge_col:
                j_all = [r for r in res if "judge" in r]
                j_acc = sum(r["judge"] for r in j_all) / len(j_all) if j_all else 0
                row += f"{j_acc:>8.4f}"
                if not is_ambiguous:
                    j_unans = [r for r, u in zip(res, unanswerable_mask) if u and "judge" in r]
                    j_u_acc = sum(r["judge"] for r in j_unans) / len(j_unans) if j_unans else 0
                    row += f"{j_u_acc:>8.4f}"
                    # placeholder for ambig col (only shows if we have ambig subset)
                    row += f"{'--':>8}"
            print(row)
        print()

    # ── Export results ──
    if output_path:
        export = {
            "metadata": {
                "strategies": strategies,
                "num_queries": len(questions),
                "top_k": top_k,
                "has_ground_truth": has_gt,
                "is_ambiguous": is_ambiguous,
                "use_judge": use_judge,
                "judge_model": _JUDGE_MODEL if use_judge else None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "results": {},
        }
        for strat in strategies:
            res = all_results[strat]
            entries = []
            for r in res:
                e = {
                    "query": r["query"],
                    "predicted_answer": r.get("predicted_answer", ""),
                    "elapsed": r["elapsed"],
                }
                if strat != "urasys":
                    e["num_results"] = len(r.get("results", []))
                else:
                    e.update({
                        "faq_answer": r.get("faq_answer", ""),
                        "doc_answer": r.get("doc_answer", ""),
                        "final_answer": r.get("final_answer", ""),
                        "faq_attempts": r.get("faq_attempts", 0),
                        "doc_attempts": r.get("doc_attempts", 0),
                    })
                if has_gt:
                    e["ground_truth"] = r.get("ground_truth", "")
                    e["is_unanswerable"] = r.get("is_unanswerable", False)
                    if "em" in r:
                        e["em"] = r["em"]
                    if "f1" in r:
                        e["f1"] = r["f1"]
                    if "judge" in r:
                        e["judge"] = r["judge"]
                entries.append(e)

            # Summary stats
            avg_t = sum(r["elapsed"] for r in res) / len(res)
            summary = {"avg_time": round(avg_t, 3), "num_queries": len(res)}
            if has_gt:
                ans_r = [r for r, u in zip(res, unanswerable_mask) if not u]
                unans_r = [r for r, u in zip(res, unanswerable_mask) if u]
                if use_em_f1 and not is_ambiguous:
                    em_r = [r for r in ans_r if "em" in r]
                    if em_r:
                        summary["em"] = round(sum(r["em"] for r in em_r) / len(em_r), 4)
                        summary["f1"] = round(sum(r["f1"] for r in em_r) / len(em_r), 4)
                j_all = [r for r in res if "judge" in r]
                if j_all:
                    summary["judge_overall"] = round(sum(r["judge"] for r in j_all) / len(j_all), 4)
                j_ans = [r for r, u in zip(res, unanswerable_mask) if not u and "judge" in r]
                if j_ans:
                    summary["judge_answerable"] = round(sum(r["judge"] for r in j_ans) / len(j_ans), 4)
                j_unans = [r for r, u in zip(res, unanswerable_mask) if u and "judge" in r]
                if j_unans:
                    summary["judge_unanswerable"] = round(sum(r["judge"] for r in j_unans) / len(j_unans), 4)
            export["results"][strat] = {"summary": summary, "queries": entries}

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        print(f"{C.GREEN}Results saved to {output_path}{C.RESET}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="URASys CLI — interactive testing & batch evaluation")
    sub = parser.add_subparsers(dest="command")

    p_int = sub.add_parser("interactive", aliases=["i"], help="Interactive REPL with live pipeline vis")
    p_int.add_argument("--strategy", choices=STRATEGIES, default="urasys")
    p_int.add_argument("--top-k", type=int, default=5)

    p_eval = sub.add_parser("eval", aliases=["e"], help="Batch evaluation (LLM-as-a-Judge by default)")
    p_eval.add_argument("--dataset", help="CSV with 'question' and optionally 'answer'/'info' columns")
    p_eval.add_argument("--queries-file", help="TSV: question<TAB>ground_truth per line")
    p_eval.add_argument("--n", type=int, default=None, help="Number of samples")
    p_eval.add_argument("--strategy", choices=STRATEGIES + ["all"], default="all")
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.add_argument("--output", help="Export to JSON (auto-generated if omitted)")
    p_eval.add_argument("--em-f1", action="store_true",
                        help="Also compute EM/F1 (only recommended for normal answerable datasets; "
                             "NOT for ambiguous/unanswerable — will be auto-ignored)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    print(f"{C.DIM}Initializing retrievers…{C.RESET}")
    _init()
    print(f"{C.GREEN}Ready.{C.RESET}\n")

    if args.command in ("interactive", "i"):
        _run_interactive(args.strategy, args.top_k)
    elif args.command in ("eval", "e"):
        questions, ground_truths = [], None
        info_items = None
        is_ambiguous = False

        if args.dataset:
            import pandas as pd
            df = pd.read_csv(ROOT / args.dataset)
            if args.n:
                df = df.head(args.n)
            if "question" not in df.columns:
                print(f"{C.RED}CSV needs 'question' column{C.RESET}")
                sys.exit(1)
            questions = df["question"].tolist()
            if "answer" in df.columns:
                ground_truths = [str(v) if pd.notna(v) else "nan" for v in df["answer"]]
            # Auto-detect ambiguous datasets (presence of 'info' column)
            if "info" in df.columns:
                is_ambiguous = True
                info_items = df["info"].tolist()
                print(f"{C.YELLOW}Detected ambiguous dataset (info column present) → LLM-as-a-Judge only{C.RESET}")
        elif args.queries_file:
            p = Path(args.queries_file)
            if not p.exists():
                print(f"{C.RED}File not found: {p}{C.RESET}")
                sys.exit(1)
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                questions.append(parts[0])
                if len(parts) > 1:
                    if ground_truths is None:
                        ground_truths = []
                    ground_truths.append(parts[1])
        else:
            print(f"{C.RED}Provide --dataset or --queries-file{C.RESET}")
            sys.exit(1)
        if not questions:
            print(f"{C.RED}No questions found.{C.RESET}")
            sys.exit(1)

        # Initialize LLM-as-a-Judge (always on by default)
        _init_judge()
        use_judge = _judge_client is not None
        if not use_judge:
            print(f"{C.RED}⚠ LLM-as-a-Judge unavailable (no OPENAI_API_KEY). Evaluation will have no metrics.{C.RESET}")
            print(f"{C.YELLOW}  Hint: set OPENAI_API_KEY or use --em-f1 for EM/F1 on answerable datasets.{C.RESET}")

        # Auto-generate output path if not specified
        output_path = args.output
        if not output_path:
            ds_name = Path(args.dataset).stem if args.dataset else "queries"
            strat_tag = args.strategy
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_path = str(ROOT / "results" / f"{ds_name}_{strat_tag}_{ts}.json")
            print(f"{C.DIM}Output → {output_path}{C.RESET}")

        use_em_f1 = getattr(args, 'em_f1', False)
        if use_em_f1 and is_ambiguous:
            print(f"{C.YELLOW}⚠ --em-f1 ignored for ambiguous datasets (LLM-as-a-Judge only){C.RESET}")
            use_em_f1 = False

        strats = STRATEGIES if args.strategy == "all" else [args.strategy]
        _run_eval(strats, questions, ground_truths, args.top_k, output_path,
                  use_judge=use_judge, is_ambiguous=is_ambiguous, info_items=info_items,
                  use_em_f1=use_em_f1)


if __name__ == "__main__":
    main()

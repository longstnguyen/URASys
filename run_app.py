import os
import sys
import json
import traceback
from pathlib import Path
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from chatbot.config.system_config import SETTINGS
from chatbot.core.model_clients import BM25Client
from chatbot.core.model_clients.embedder.openai import OpenAIClientConfig, OpenAIEmbedder
from chatbot.core.model_clients.llm.google import GoogleAIClientLLMConfig, GoogleAIClientLLM
from chatbot.core.retriever import DocumentRetriever, FAQRetriever
from chatbot.core.retriever.base_class import DocumentRetrievalResult, FAQRetrievalResult
from chatbot.indexing.context_document.base_class import PreprocessingConfig
from chatbot.indexing.faq.base_class import FAQDocument
from chatbot.workflow.build_index import DataIndex
from chatbot.utils.base_class import ModelsConfig
from chatbot.utils.database_clients.milvus import MilvusVectorDatabase, MilvusConfig


class DocumentRetrievalOutput(BaseModel):
    status: str
    results: Optional[dict] = None
    message: Optional[str] = None


class FAQRetrievalOutput(BaseModel):
    status: str
    results: Optional[dict] = None
    message: Optional[str] = None


class IndexingOutput(BaseModel):
    status: str
    message: str
    file_count: int = 0


class ServerStatus(BaseModel):
    status: str
    message: str


app = FastAPI(title="URASys - Unified RAG System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = None
vector_db = None
llm = None
document_retriever = None
faq_retriever = None
document_bm25_client = None
faq_bm25_client = None
preprocessing_config = None
models_config = None


@app.on_event("startup")
async def startup_event():
    global embedder, vector_db, llm, document_retriever, faq_retriever
    global document_bm25_client, faq_bm25_client, preprocessing_config, models_config
    
    try:
        logger.info("Starting URASys - Unified RAG System...")
        
        config_path = Path(__file__).parent / "chatbot/config/models_config.json"
        with open(config_path, "r") as f:
            models_config = ModelsConfig.from_dict(json.load(f))
        
        logger.info("Initializing embedder...")
        embedder = OpenAIEmbedder(config=OpenAIClientConfig(
            api_key=SETTINGS.OPENAI_API_KEY,
            model=models_config.embedding_config.model_id
        ))
        
        logger.info("Initializing vector database...")
        vector_db = MilvusVectorDatabase(
            config=MilvusConfig(
                cloud_uri=SETTINGS.MILVUS_CLOUD_URI,
                token=SETTINGS.MILVUS_CLOUD_TOKEN,
                run_async=False
            )
        )
        
        logger.info("Initializing LLM...")
        llm = GoogleAIClientLLM(
            config=GoogleAIClientLLMConfig(
                api_key=SETTINGS.GEMINI_API_KEY,
                model=models_config.llm_config["indexing_llm"].model_id,
                temperature=models_config.llm_config["indexing_llm"].temperature,
                max_tokens=models_config.llm_config["indexing_llm"].max_new_tokens,
                thinking_budget=1000,
            )
        )
        
        preprocessing_config = PreprocessingConfig()
        
        init_document_bm25()
        init_faq_bm25()
        
        init_document_retriever()
        init_faq_retriever()
        
        logger.info("URASys initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.error(traceback.format_exc())


def init_document_bm25() -> bool:
    global document_bm25_client
    
    try:
        bm25_path = Path(__file__).parent / "chatbot/data/bm25/document/state_dict.json"
        if not bm25_path.exists():
            logger.warning("Document BM25 data not found")
            document_bm25_client = BM25Client(language="en", init_without_load=True)
            return False
        
        document_bm25_client = BM25Client(
            language="en",
            local_path=str(bm25_path),
            init_without_load=False
        )
        logger.info("Document BM25 client initialized")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing document BM25: {e}")
        document_bm25_client = BM25Client(language="en", init_without_load=True)
        return False


def init_faq_bm25() -> bool:
    global faq_bm25_client
    
    try:
        bm25_path = Path(__file__).parent / "chatbot/data/bm25/faq/state_dict.json"
        if not bm25_path.exists():
            logger.warning("FAQ BM25 data not found")
            faq_bm25_client = BM25Client(language="en", init_without_load=True)
            return False
        
        faq_bm25_client = BM25Client(
            language="en",
            local_path=str(bm25_path),
            init_without_load=False
        )
        logger.info("FAQ BM25 client initialized")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing FAQ BM25: {e}")
        faq_bm25_client = BM25Client(language="en", init_without_load=True)
        return False


def init_document_retriever() -> bool:
    global document_retriever
    
    try:
        if document_bm25_client is None:
            return False
        
        document_retriever = DocumentRetriever(
            collection_name="document_collection",
            embedder=embedder,
            bm25_client=document_bm25_client,
            vector_db=vector_db
        )
        logger.info("Document retriever initialized")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing document retriever: {e}")
        return False


def init_faq_retriever() -> bool:
    global faq_retriever
    
    try:
        if faq_bm25_client is None:
            return False
        
        faq_retriever = FAQRetriever(
            collection_name="faq_collection",
            embedder=embedder,
            bm25_client=faq_bm25_client,
            vector_db=vector_db
        )
        logger.info("FAQ retriever initialized")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing FAQ retriever: {e}")
        return False


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "embedder": embedder is not None,
            "vector_db": vector_db is not None,
            "llm": llm is not None,
            "document_retriever": document_retriever is not None,
            "faq_retriever": faq_retriever is not None,
        }
    }


@app.get("/")
async def root():
    return {
        "name": "URASys - Unified RAG System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "document_retrieval": "/documents/retrieve",
            "faq_retrieval": "/faq/retrieve",
            "index_documents": "/index/documents",
            "index_faq": "/index/faq"
        }
    }


@app.post("/documents/retrieve", response_model=DocumentRetrievalOutput)
async def retrieve_documents(
    query: str,
    top_k: int = 5
):
    try:
        if document_retriever is None:
            # Try to initialize
            init_document_retriever()
            if document_retriever is None:
                raise HTTPException(
                    status_code=503,
                    detail="Document retriever not available. Please index documents first."
                )
        
        results = document_retriever.retrieve_documents(
            query=query,
            top_k=top_k
        )
        
        return DocumentRetrievalOutput(
            status="success",
            results=results.model_dump() if results else None
        )
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        logger.error(traceback.format_exc())
        return DocumentRetrievalOutput(
            status="error",
            message=str(e)
        )


@app.get("/documents/status")
async def document_retriever_status():
    """Check document retriever status"""
    return ServerStatus(
        status="ready" if document_retriever is not None else "not_ready",
        message="Document retriever is ready" if document_retriever is not None 
                else "Document retriever not initialized. Please index documents first."
    )


@app.post("/faq/retrieve", response_model=FAQRetrievalOutput)
async def retrieve_faqs(
    query: str,
    top_k: int = 5
):
    try:
        if faq_retriever is None:
            init_faq_retriever()
            if faq_retriever is None:
                raise HTTPException(
                    status_code=503,
                    detail="FAQ retriever not available. Please index FAQs first."
                )
        
        results = faq_retriever.retrieve_faqs(
            query=query,
            top_k=top_k
        )
        
        return FAQRetrievalOutput(
            status="success",
            results=results.model_dump() if results else None
        )
        
    except Exception as e:
        logger.error(f"Error retrieving FAQs: {e}")
        logger.error(traceback.format_exc())
        return FAQRetrievalOutput(
            status="error",
            message=str(e)
        )


@app.get("/faq/status")
async def faq_retriever_status():
    return ServerStatus(
        status="ready" if faq_retriever is not None else "not_ready",
        message="FAQ retriever is ready" if faq_retriever is not None 
                else "FAQ retriever not initialized. Please index FAQs first."
    )


@app.post("/index/documents", response_model=IndexingOutput)
async def index_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    try:
        documents = []
        for file in files:
            content = await file.read()
            documents.append(content.decode('utf-8'))
        
        indexer = DataIndex(
            llm=llm,
            embedder=embedder,
            document_bm25_client=document_bm25_client,
            faq_bm25_client=faq_bm25_client,
            preprocessing_config=preprocessing_config,
            vector_db=vector_db
        )
        
        indexer.run_insert(
            documents=documents,
            faqs=[],
            document_collection_name="document_collection",
            faq_collection_name="faq_collection"
        )
        
        bm25_dir = Path(__file__).parent / "chatbot/data/bm25/document"
        bm25_dir.mkdir(parents=True, exist_ok=True)
        document_bm25_client.save(path=str(bm25_dir / "state_dict.json"), auto_save_local=True)
        
        init_document_bm25()
        init_document_retriever()
        
        return IndexingOutput(
            status="success",
            message="Documents indexed successfully",
            file_count=len(files)
        )
        
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        logger.error(traceback.format_exc())
        return IndexingOutput(
            status="error",
            message=str(e)
        )


@app.post("/index/faq", response_model=IndexingOutput)
async def index_faq(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    try:
        import pandas as pd
        
        content = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(content))
        
        faqs = [
            FAQDocument(question=row['question'], answer=row['answer'])
            for _, row in df.iterrows()
        ]
        
        indexer = DataIndex(
            llm=llm,
            embedder=embedder,
            document_bm25_client=document_bm25_client,
            faq_bm25_client=faq_bm25_client,
            preprocessing_config=preprocessing_config,
            vector_db=vector_db
        )
        
        indexer.run_insert(
            documents=[],
            faqs=faqs,
            document_collection_name="document_collection",
            faq_collection_name="faq_collection"
        )
        
        bm25_dir = Path(__file__).parent / "chatbot/data/bm25/faq"
        bm25_dir.mkdir(parents=True, exist_ok=True)
        faq_bm25_client.save(path=str(bm25_dir / "state_dict.json"), auto_save_local=True)
        
        init_faq_bm25()
        init_faq_retriever()
        
        return IndexingOutput(
            status="success",
            message="FAQ indexed successfully",
            file_count=1
        )
        
    except Exception as e:
        logger.error(f"Error indexing FAQ: {e}")
        logger.error(traceback.format_exc())
        return IndexingOutput(
            status="error",
            message=str(e)
        )


def main():
    logger.info("=" * 60)
    logger.info("Starting URASys - Unified RAG Application System")
    logger.info("Simple single-process mode (No Docker)")
    logger.info("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()

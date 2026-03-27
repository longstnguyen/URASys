import os
from typing import Optional

import json
import traceback
from asyncio import Lock
from pydantic import BaseModel, Field
from fastmcp import FastMCP
from loguru import logger
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
import uvicorn

from urasys.config.system_config import SETTINGS
from urasys.config.utils import get_milvus_config
from urasys.core.model_clients import BM25Client
from urasys.core.model_clients.embedder.openai import OpenAIClientConfig, OpenAIEmbedder
from urasys.core.retriever import FAQRetriever
from urasys.core.retriever.base_class import FAQRetrievalResult
from urasys.utils.base_class import ModelsConfig
from urasys.utils.database_clients.milvus import MilvusVectorDatabase, MilvusConfig

# ------------------- Global Model Classes -------------------

class FAQRetrievalOutput(BaseModel):
    """Class to store FAQ retrieval response data."""
    status: str = Field(..., description="The status of the retrieval operation.")
    results: Optional[FAQRetrievalResult] = Field(None, description="The retrieved FAQs.")
    message: Optional[str] = Field(None, description="An error message if the operation failed.")


class ServerStatus(BaseModel):
    """Class to store server status response data."""
    status: str = Field(..., description="The status of the server.")
    message: str = Field(..., description="A message providing additional information about the server status.")


# ------------------- Init Server API -------------------

# Create a lock to ensure thread safety when accessing shared resources
rw_lock = Lock()

mcp = FastMCP(
    name="FAQ Retrieval Server",
    instructions="This is a server for retrieving relevant FAQs based on user queries.",
    include_tags=["faq"]
)

# Global variables - initialized as None
retriever = None
embedder = None
vector_db = None

try:
    # Load the FAQ retrieval model
    logger.info("Starting up FAQ Retrieval server...")
    
    models_config = {}
    with open("./src/urasys/config/models_config.json", "r") as f:
        # Load the JSON file
        models_config = json.load(f)

        # Convert the loaded JSON to a ModelsConfig object
        embedder_config = ModelsConfig.from_dict(models_config).embedding_config

    # Initialize the embedder
    embedder = OpenAIEmbedder(config=OpenAIClientConfig(
        api_key=SETTINGS.OPENAI_API_KEY,
        model=embedder_config.model_id
    ))

    # Initialize the vector database client
    vector_db = MilvusVectorDatabase(
        config=get_milvus_config(SETTINGS, run_async=False)
    )
    
    logger.info("FAQ Retrieval Server base components initialized successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.error(traceback.format_exc())

def check_retriever_ready() -> bool:
    """Check if retriever is ready for queries."""
    return retriever is not None

def initialize_retriever() -> bool:
    """Initialize retriever with BM25 weights. Returns True if successful."""
    global retriever
    
    try:
        # Check if BM25 state dict exists
        bm25_path = "./src/urasys/data/bm25/faq/state_dict.json"
        if not os.path.exists(bm25_path):
            logger.warning(f"BM25 state dict not found at {bm25_path}")
            return False

        # Initialize BM25 client
        bm25_client = BM25Client(
            local_path=bm25_path,
            init_without_load=False
        )

        # Initialize retriever
        retriever = FAQRetriever(
            collection_name=SETTINGS.MILVUS_COLLECTION_FAQ_NAME,
            embedder=embedder,
            bm25_client=bm25_client,
            vector_db=vector_db
        )
        
        logger.info("FAQ retriever initialized successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing retriever: {e}")
        logger.error(traceback.format_exc())
        return False

async def manual_initialize(request: Request):
    """Manually initialize the retriever."""
    async with rw_lock:
        try:
            if initialize_retriever():
                return JSONResponse(
                    content={
                        "status": "success",
                        "message": "FAQ retriever initialized successfully."
                    },
                    status_code=200
                )
            else:
                return JSONResponse(
                    content={
                        "status": "error",
                        "message": "Failed to initialize FAQ retriever. Check if BM25 weights are available."
                    },
                    status_code=500
                )
        except Exception as e:
            logger.error(f"Error during manual initialization: {e}")
            return JSONResponse(
                content={
                    "status": "error",
                    "message": f"Manual initialization failed: {str(e)}"
                },
                status_code=500
            )
# ------------------- API Endpoints -------------------

@mcp.tool(
    name="faq_retrieval_tool",
    description="Retrieve top K relevant FAQs based on the query.",
    tags=["faq"]
)
async def retrieve(
    query: str = Field(..., description="The query string for FAQ retrieval."),
    top_k: int = Field(5, description="The number of top FAQs to retrieve.")
) -> FAQRetrievalOutput:
    async with rw_lock:
        try:
            # Check if retriever is ready
            if not check_retriever_ready():
                # Try to initialize retriever
                if not initialize_retriever():
                    return FAQRetrievalOutput(
                        status="error", 
                        message=(
                            "Retriever not ready. Please ensure indexing has "
                            "been completed and weights are available."
                        )
                    )
                
            # Retrieve relevant FAQs
            results = retriever.retrieve_faqs(
                query=query,
                top_k=top_k
            )
            return FAQRetrievalOutput(status="success", results=results)
        except Exception as e:
            logger.error(f"Error retrieving relevant FAQs: {e}")
            logger.error(traceback.format_exc())
            return FAQRetrievalOutput(status="error", message=str(e))
    

async def health_check(request: Request):
    """Health check endpoint to verify the server is running."""
    retriever_status = "ready" if check_retriever_ready() else "not_ready"

    return JSONResponse(
        content={
            "status": "healthy",
            "message": f"FAQ Retrieval Server is running. Retriever status: {retriever_status}"
        },
        status_code=200
    )
    
async def reload_index(request: Request):
    """Reload the FAQ index."""
    async with rw_lock:
        try:
            # Load the BM25 state dict from local path
            logger.info("Reloading FAQ index from local path...")

            # Store reference to old client for cleanup
            global retriever
            old_bm25_client = retriever.bm25_client if retriever else None

            # Initialize new BM25 models
            new_bm25_client = BM25Client(
                local_path="./src/urasys/data/bm25/faq/state_dict.json",
                init_without_load=False
            )

            # Reinitialize the retriever with the new BM25 client
            retriever = FAQRetriever(
                collection_name=SETTINGS.MILVUS_COLLECTION_FAQ_NAME,
                embedder=embedder,
                bm25_client=new_bm25_client,
                vector_db=vector_db
            )

            # Cleanup old client if exists
            if old_bm25_client:
                del old_bm25_client
                logger.info("Old BM25 client cleaned up")

            logger.info("FAQ index reloaded successfully")

            return JSONResponse(
                content={
                    "status": "success",
                    "message": "FAQ index reloaded successfully."
                },
                status_code=200
            )
        except Exception as e:
            logger.error(f"Error reloading FAQ index: {e}")
            return JSONResponse(
                content={
                    "status": "error",
                    "message": str(e)
                },
                status_code=500
            )
    
# ------------------- MCP Server Check -------------------

async def check_mcp(mcp: FastMCP):
    tools = mcp.list_tools()
    resources = mcp.list_resources()
    templates = mcp.list_resource_templates()
    logger.info(
        f"MCP server ready — Tools: {len(tools)}, Resources: {len(resources)}, "
        f"Templates: {len(templates)}"
    )


def main():
    import asyncio
    asyncio.run(check_mcp(mcp))
    app = Starlette(
        routes=[
            Route("/initialize", endpoint=manual_initialize, methods=["POST"]),
            Route("/health", endpoint=health_check, methods=["GET"]),
            Route("/reload-index", endpoint=reload_index, methods=["POST"]),
            Mount("/", app=mcp.sse_app()),
        ]
    )
    uvicorn.run(app, host="0.0.0.0", port=8011)


if __name__ == "__main__":
    main()

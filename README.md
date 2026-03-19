# URASys - Unified RAG Application System

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Google ADK](https://img.shields.io/badge/Google_ADK-Latest-red.svg)](https://github.com/google/genai-agent-dev-kit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
---

## Overview

URASys is a production-ready Retrieval-Augmented Generation (RAG) system that combines the power of **multi-agent architecture**, **hybrid search**, and **advanced reasoning** to provide accurate, context-aware responses from your knowledge base.

### Why URASys?

- **Multi-Agent Architecture**: Parallel FAQ and Document search agents coordinated by an intelligent manager agent
- **Hybrid Search**: Combines BM25 lexical search with dense vector embeddings for optimal retrieval
- **Agent Reasoning**: Built on Google ADK with sophisticated query decomposition and result synthesis
- **Bilingual Support**: Optimized for both English and Vietnamese queries
- **Document Intelligence**: Semantic chunking and context-aware document processing
- **Production Ready**: FastAPI backend with MCP (Model Context Protocol) servers

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent Coordination** | Manager agent orchestrates parallel FAQ and Document search agents for comprehensive results |
| **Hybrid Retrieval** | BM25 + Vector embeddings for superior search accuracy |
| **Semantic Chunking** | Intelligent document segmentation preserving context and meaning |
| **Context Augmentation** | Automatic FAQ generation and document context extraction |
| **Conversational Memory** | Maintains dialogue context for natural follow-up questions |
| **Query Decomposition** | Breaks complex queries into focused sub-queries |
| **Result Synthesis** | Aggregates and ranks results from multiple sources |
| **Bilingual Processing** | Native support for Vietnamese and English |


---

## Quick Start

### Prerequisites

- Python 3.13+
- OpenAI API Key
- Google Gemini API Key
- Milvus Cloud Account (free tier available at [zilliz.com](https://zilliz.com/cloud))

### 1. Clone and Setup Environment

```bash
# Clone repository
git clone https://github.com/quin210/urasys.git
cd urasys

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create `.env` file in the project root:

```bash
# API Keys
OPENAI_API_KEY=sk-your-openai-key
GEMINI_API_KEY=your-gemini-key

# Milvus Cloud Configuration
MILVUS_CLOUD_URI=https://your-cluster.api.gcp-us-west1.zillizcloud.com
MILVUS_CLOUD_TOKEN=your-milvus-token

# Optional: Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gemini-2.5-flash
```

### 4. Run the Application
To run this command, you need to run 4 different terminal commands in parallel:
**Terminal 1 - FAQ MCP Server:**
```bash
source .venv/bin/activate
python -m chatbot.server.faq_server.server_app
# Running on http://localhost:8001
```

**Terminal 2 - Document MCP Server:**
```bash
source .venv/bin/activate
python -m chatbot.server.document_server.server_app
# Running on http://localhost:8002
```

**Terminal 3 - MCP Sever:**
```bash
source .venv/bin/activate
python run_app.py
```

**Terminal 4 -  ADK Web Interface:**
```bash
source .venv/bin/activate
adk web --port 8010 --host 0.0.0.0 --reload chatbot/core/
```

**Done!** Access [http://localhost:8000/docs](http://localhost:8000/docs) to explore the API.

---
### 5. Access the System

- **ADK Web UI**: http://localhost:8010
- **API Documentation**: http://localhost:8000/docs (if using `run_app.py`)

---

## Usage Examples

### Example 1: Simple Query

```python
# Query via API
import requests

response = requests.post(
    "http://localhost:8010/run_sse",
    json={
        "appName": "agents",
        "newMessage": "What are the admission requirements?"
    }
)
```

### Example 2: Follow-up Questions

The system maintains conversation context:

```
User: "What are the scholarship options?"
Agent: [Lists scholarship programs A, B, C...]

User: "Tell me more about program B"
Agent: [Provides detailed info about program B, understanding context]
```

### Example 3: Complex Query

```
User: "Compare the engineering and business programs, 
       including admission requirements and career prospects"
       
Agent: [Decomposes into sub-queries, searches both domains, 
        synthesizes comprehensive comparison]
```

---

## Project Structure

```
urasys/
├── chatbot/
│   ├── config/                    # System configuration
│   │   ├── models_config.json     # Model settings
│   │   └── system_config.py       # Environment config
│   │
│   ├── core/                      # Core functionality
│   │   ├── agents/                # Multi-agent system
│   │   │   ├── agent.py           # Manager agent
│   │   │   ├── tools.py           # Agent tools
│   │   │   ├── prompt.py          # Agent prompts
│   │   │   └── sub_agents/        # Specialized agents
│   │   │       ├── faq_search_agent/
│   │   │       └── document_search_agent/
│   │   │
│   │   ├── model_clients/         # LLM & Embedding clients
│   │   │   ├── llm/               # Language model clients
│   │   │   ├── embedder/          # Embedding clients
│   │   │   └── bm25.py            # BM25 implementation
│   │   │
│   │   └── retriever/             # Retrieval systems
│   │       ├── faq_retriever.py
│   │       └── document_retriever.py
│   │
│   ├── indexing/                  # Document processing
│   │   ├── context_document/      # Document chunking & processing
│   │   │   ├── semantic_chunk.py  # Semantic chunking
│   │   │   ├── extract_context.py # Context extraction
│   │   │   └── augment_context.py # Context augmentation
│   │   │
│   │   └── faq/                   # FAQ processing
│   │       ├── generate_document.py
│   │       ├── expand_document.py
│   │       └── augment_document.py
│   │
│   ├── server/                    # MCP Servers
│   │   ├── faq_server/            # FAQ retrieval server (port 8001)
│   │   ├── document_server/       # Document retrieval server (port 8002)
│   │   └── index_server/          # Indexing server
│   │
│   ├── prompts/                   # Prompt templates
│   │   ├── indexing/              # Indexing prompts
│   │   └── query/                 # Query prompts
│   │
│   ├── utils/                     # Utilities
│   │   ├── database_clients/      # DB clients
│   │   │   ├── milvus/            # Milvus implementation
│   │   │   └── lancedb/           # LanceDB implementation
│   │   └── embeddings.py          # Embedding utilities
│   │
│   └── workflow/                  # Workflows
│       ├── build_index.py         # Indexing workflow
│       └── chatbot_inference.py   # Inference workflow
│
├── dataset/                       # Sample datasets
├── environments/                  # Environment configs
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies
├── run_app.py                    # Main application
└── README.md                     # This file
```

---

## Configuration

### Model Configuration

Edit `chatbot/config/models_config.json`:

```json
{
  "embedder": {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "dimension": 1536
  },
  "llm": {
    "provider": "google",
    "model": "gemini-2.5-flash",
    "temperature": 0.2
  },
  "retrieval": {
    "top_k": 5,
    "similarity_threshold": 0.7
  }
}
```

### System Configuration

Customize `chatbot/config/system_config.py` for:
- Database collection names
- BM25 parameters
- Chunking strategies
- Agent retry limits


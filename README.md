# URASys — Unified Retrieval Agent System

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![CopilotKit](https://img.shields.io/badge/CopilotKit-latest-purple.svg)](https://copilotkit.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

URASys is a multi-agent Retrieval-Augmented Generation (RAG) system that handles **ambiguous and unanswerable questions** through sophisticated iterative retrieval. It combines a Manager LLM with two specialised Sub-Agents (FAQ and Document), hybrid BM25+dense retrieval, and RRF fusion.

---

## Architecture

```
User Query
    │
    ▼
Manager Agent (Gemini 2.5 Flash)
    │  Query Decomposition → sub-queries
    ▼
┌─────────────────────────────────────────┐
│  For each sub-query (parallel):         │
│                                         │
│  FAQ Sub-Agent          Doc Sub-Agent   │
│  ┌────────────┐         ┌────────────┐  │
│  │ LLM loop   │         │ LLM loop   │  │
│  │ faq_tool   │         │ doc_tool   │  │
│  │ (≤3 tries) │         │ (≤3 tries) │  │
│  └─────┬──────┘         └─────┬──────┘  │
│        │ Hybrid RRF           │          │
│        │ (BM25 + Dense)       │          │
│        ▼                      ▼          │
│  grounded text           grounded text  │
└─────────────────────────────────────────┘
    │
    ▼
Manager synthesises PATH A/B/C/D response
```

**PATH A** — Direct answer found  
**PATH B** — Query too broad → ask clarifying question  
**PATH C** — Vague query → refine and retry (up to 4 Manager attempts)  
**PATH D** — Off-topic or exhausted → honest no-answer  

---

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15 + CopilotKit |
| Manager Agent | CopilotKit `useCopilotAction` + Gemini 2.5 Flash |
| Sub-Agents | `google-genai` SDK, iterative tool-call loops |
| Retrieval | Milvus (dense) + BM25 → RRF fusion |
| Backend API | FastAPI (port 8005) |
| Embedding | OpenAI `text-embedding-3-small` |

---

## Quick Start

### Prerequisites

- Python 3.13+
- Node.js 18+
- OpenAI API Key  
- Google Gemini API Key  
- Milvus Cloud account ([zilliz.com](https://zilliz.com/cloud))

### 1. Setup Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configure Environment Variables

Create `environments/.env`:

```bash
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
MILVUS_CLOUD_URI=https://your-cluster.api.gcp-us-west1.zillizcloud.com
MILVUS_CLOUD_TOKEN=...
MILVUS_COLLECTION_FAQ_NAME=faq_data
MILVUS_COLLECTION_DOCUMENT_NAME=document_data
```

### 3. Build the Index

```bash
PYTHONPATH=src python -m urasys.workflow.build_index
```

### 4. Start the Services

Run each in a separate terminal:

**Terminal 1 — FAQ MCP Server (port 8011):**
```bash
PYTHONPATH=src python -m urasys.server.faq_server.server_app
```

**Terminal 2 — Document MCP Server (port 8012):**
```bash
PYTHONPATH=src python -m urasys.server.document_server.server_app
```

**Terminal 3 — CopilotKit Backend (port 8005):**
```bash
PYTHONPATH=src python -m urasys.server.copilotkit_server.server_app
```

**Terminal 4 — Next.js Frontend (port 3000):**
```bash
cd frontend && npm install && npm run dev
```

Open **http://localhost:3000**

---

## Project Structure

```
URASys/
├── src/urasys/
│   ├── config/                    # System & model configuration
│   ├── core/
│   │   ├── agents/                # Manager agent (legacy ADK wrappers — unused)
│   │   ├── model_clients/         # LLM, Embedder, BM25 clients
│   │   └── retriever/             # FAQ & Document retrievers (hybrid RRF)
│   ├── indexing/                  # Document chunking, FAQ generation
│   ├── prompts/
│   │   ├── indexing/              # Prompts for index-time operations
│   │   └── query/                 # Sub-agent system prompts (paper Fig. 3/4)
│   ├── server/
│   │   ├── faq_server/            # MCP FAQ retrieval (port 8011)
│   │   ├── document_server/       # MCP document retrieval (port 8012)
│   │   ├── copilotkit_server/     # Main backend + sub-agent LLM loops (port 8005)
│   │   └── index_server/          # Index management
│   ├── utils/                     # DB clients (Milvus/LanceDB), embeddings
│   └── workflow/                  # build_index, chatbot_inference
├── frontend/                      # Next.js + CopilotKit chat UI
├── datasets/                      # Evaluation datasets
├── scripts/                       # Utility scripts
└── environments/                  # .env files
```

---

## Key Design Decisions

- **Sub-Agents are LLM loops, not simple retrievers.** Each sub-agent issues tool calls iteratively (up to 3 rounds), reformulating its search query if results are unsatisfactory. This matches the paper's Algorithm 1.
- **Manager only sees grounded text.** The `faq_answer` / `doc_answer` fields from sub-agents are what the Manager uses to decide PATH A/B/C/D — not raw chunk lists.
- **No ADK dependency.** The system was originally built with Google ADK; inference now runs via `google-genai` SDK directly with CopilotKit managing the Manager Agent turn.


"""
Offline indexing script - index datasets directly without starting a server.

Usage:
    python scripts/index_offline.py \
        --dataset datasets/Hotpot_1000.csv \
        --n 5 \
        --mode index

Arguments:
    --dataset   Path to CSV dataset (relative to repo root)
    --n         Number of samples to index (default: all)
    --mode      "index" (overwrite) or "insert" (append)
    --doc-collection    Milvus document collection name (default: document_data)
    --faq-collection    Milvus FAQ collection name (default: faq_data)

Expected CSV columns:
    - paragraph: document text (plain string or list-of-dicts from HotpotQA)
    - question:  question string
    - answer:    answer string
"""
import ast
import json
import os
import sys
import time
import logging
import warnings
import argparse
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Suppress noisy loggers before importing project modules
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore", message=".*PyTorch.*")
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

import pandas as pd
from loguru import logger

# Suppress loguru noise — rich handles all output
logger.remove()
logger.add(sys.stderr, level="ERROR")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich import box

from urasys.config.system_config import SETTINGS
from urasys.config.utils import get_milvus_config
from urasys.core.model_clients import BM25Client
from urasys.core.model_clients.embedder.openai import OpenAIClientConfig, OpenAIEmbedder
from urasys.core.model_clients.llm.google import GoogleAIClientLLMConfig, GoogleAIClientLLM
from urasys.indexing.context_document.base_class import PreprocessingConfig
from urasys.indexing.faq.base_class import FAQDocument
from urasys.workflow.build_index import DataIndex
from urasys.utils.base_class import ModelsConfig
from urasys.utils.database_clients.milvus import MilvusVectorDatabase, MilvusConfig

console = Console()


def parse_paragraph(raw) -> str:
    """Handle HotpotQA paragraph column which is a list-of-dicts."""
    try:
        items = ast.literal_eval(raw) if isinstance(raw, str) else raw
        if isinstance(items, list):
            return " ".join(
                item.get("paragraph_text", "") for item in items if isinstance(item, dict)
            )
    except Exception:
        pass
    return str(raw)


def main():
    parser = argparse.ArgumentParser(description="Offline indexing for URASys")
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset (relative to repo root)")
    parser.add_argument("--n", type=int, default=None, help="Number of samples (default: all)")
    parser.add_argument("--mode", choices=["index", "insert"], default="index",
                        help="'index' overwrites collection, 'insert' appends")
    parser.add_argument("--doc-collection", default="document_data")
    parser.add_argument("--faq-collection", default="faq_data")
    args = parser.parse_args()

    t_start = time.time()
    root = Path(__file__).parent.parent

    # ── Banner ──────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold white]URASys Offline Indexing[/bold white]\n"
        "[dim]Semantic chunking • Context extraction • FAQ generation • Full pipeline[/dim]",
        border_style="magenta", padding=(1, 3),
    ))
    console.print()

    # ── Step 1: Load dataset ────────────────────────────
    console.print(Rule("[bold magenta]Step 1/4[/bold magenta]  Load Dataset", style="magenta"))
    dataset_path = root / args.dataset
    df = pd.read_csv(dataset_path)
    if args.n is not None:
        df = df.head(args.n)

    para_col = next((c for c in ["paragraph", "paragraphs"] if c in df.columns), None)
    if para_col is None:
        raise ValueError(f"No paragraph column found. Available: {df.columns.tolist()}")

    documents = [parse_paragraph(row[para_col]) for _, row in df.iterrows()]

    faqs = []
    if "question" in df.columns and "answer" in df.columns:
        faqs = [
            FAQDocument(
                id=str(uuid.uuid4()),
                question=str(row["question"]),
                answer=str(row["answer"])
            )
            for _, row in df.iterrows()
        ]

    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_column(style="dim")
    info.add_column(style="bold white")
    info.add_row("Dataset", str(dataset_path.name))
    info.add_row("Samples", str(len(df)))
    info.add_row("Documents", str(len(documents)))
    info.add_row("FAQs", str(len(faqs)))
    info.add_row("Mode", args.mode)
    info.add_row("Doc collection", args.doc_collection)
    info.add_row("FAQ collection", args.faq_collection)
    console.print(info)
    console.print()

    # ── Step 2: Initialize components ───────────────────
    console.print(Rule("[bold magenta]Step 2/4[/bold magenta]  Initialize Components", style="magenta"))

    config_path = root / "src/urasys/config/models_config.json"
    with open(config_path) as f:
        models_config = ModelsConfig.from_dict(json.load(f))

    with console.status("[bold magenta]Connecting embedder, LLM, Milvus…"):
        embedder = OpenAIEmbedder(config=OpenAIClientConfig(
            api_key=SETTINGS.OPENAI_API_KEY,
            model=models_config.embedding_config.model_id
        ))

        vector_db = MilvusVectorDatabase(
            config=get_milvus_config(SETTINGS, run_async=False)
        )

        llm = GoogleAIClientLLM(
            config=GoogleAIClientLLMConfig(
                api_key=SETTINGS.GEMINI_API_KEY,
                model=models_config.llm_config["indexing_llm"].model_id,
                temperature=models_config.llm_config["indexing_llm"].temperature,
                max_tokens=models_config.llm_config["indexing_llm"].max_new_tokens,
                thinking_budget=1000,
            )
        )

    document_bm25_client = BM25Client(language="en", init_without_load=True)
    faq_bm25_client = BM25Client(language="en", init_without_load=True)

    indexer = DataIndex(
        llm=llm,
        embedder=embedder,
        document_bm25_client=document_bm25_client,
        faq_bm25_client=faq_bm25_client,
        preprocessing_config=PreprocessingConfig(),
        vector_db=vector_db
    )

    console.print(f"  [green]✓[/green] Embedder: [bold]{models_config.embedding_config.model_id}[/bold]")
    console.print(f"  [green]✓[/green] LLM: [bold]{models_config.llm_config['indexing_llm'].model_id}[/bold]")
    console.print(f"  [green]✓[/green] Milvus connected")
    console.print()

    # ── Step 3: Build index (LLM pipeline) ──────────────
    console.print(Rule("[bold magenta]Step 3/4[/bold magenta]  Build Index (LLM Pipeline)", style="magenta"))
    console.print("  [dim]Semantic chunking → Context extraction → Chunk reconstruction → FAQ generation[/dim]")
    console.print()

    # ── Step 4: Index into Milvus ───────────────────────
    # (Steps 3 & 4 are handled inside DataIndex — the build_index module
    #  prints its own rich progress. We just call and show summary.)

    if args.mode == "index":
        index_data = indexer.run_index(
            documents=documents,
            faqs=faqs,
            document_collection_name=args.doc_collection,
            faq_collection_name=args.faq_collection
        )
    else:
        index_data = indexer.run_insert(
            documents=documents,
            faqs=faqs,
            document_collection_name=args.doc_collection,
            faq_collection_name=args.faq_collection
        )

    console.print()
    console.print(Rule("[bold magenta]Step 4/4[/bold magenta]  Done", style="magenta"))

    # ── Summary ─────────────────────────────────────────
    elapsed = time.time() - t_start
    summary = Table(title="URASys Indexing Complete", box=box.ROUNDED,
                    title_style="bold green", border_style="green")
    summary.add_column("Metric", style="dim")
    summary.add_column("Value", style="bold white")
    summary.add_row("Documents processed", str(len(documents)))
    summary.add_row("Chunks indexed", str(len(index_data.documents)))
    summary.add_row("FAQs indexed", str(len(index_data.faqs)))
    summary.add_row("Doc collection", args.doc_collection)
    summary.add_row("FAQ collection", args.faq_collection)
    summary.add_row("Total time", f"{elapsed:.1f}s")
    console.print(summary)
    console.print()


if __name__ == "__main__":
    main()

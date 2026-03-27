"""
Baseline indexing — simple chunking + encoding, no LLM involved.

This creates a simple baseline index for comparison with URASys.
Only chunks documents with fixed-size window — no semantic chunking,
no context extraction, no FAQ generation. Plain RAG.

Uses the SAME Milvus document schema as URASys but stores in a
SEPARATE collection (default: baseline_document_data).

Usage:
    # Index 100 samples from ViQuAD2 (overwrites)
    python scripts/index_baseline.py \\
        --dataset datasets/ViQuAD2_1000.csv --n 100

    # Custom collection name + chunk size
    python scripts/index_baseline.py \\
        --dataset datasets/ViQuAD2_1000.csv --n 100 \\
        --doc-collection baseline_doc \\
        --chunk-size 500 --chunk-overlap 50

    # Append mode
    python scripts/index_baseline.py \\
        --dataset datasets/ViQuAD2_1000.csv --n 50 --mode insert
"""
import ast
import json
import sys
import time
import logging
import warnings
import argparse
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import os
if not os.environ.get("ENVIRONMENT_FILE"):
    os.environ["ENVIRONMENT_FILE"] = str(ROOT / "environments" / ".env")

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
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
)
from rich.rule import Rule
from rich import box

from urasys.config.system_config import SETTINGS
from urasys.config.utils import get_milvus_config
from urasys.core.model_clients import BM25Client
from urasys.core.model_clients.embedder.openai import OpenAIClientConfig, OpenAIEmbedder
from urasys.utils.base_class import ModelsConfig
from urasys.utils.database_clients.milvus import MilvusVectorDatabase
from urasys.utils.vectordb_schema import (
    DOCUMENT_DATABASE_SCHEMA,
)

console = Console()

def _progress_bar(description: str = "") -> Progress:
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="green", finished_style="bold green"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )


# ── Simple fixed-size chunking ────────────────────────────

def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into fixed-size character chunks with overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


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


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline indexing — simple chunking, no LLM")
    parser.add_argument("--dataset", required=True, help="CSV dataset (relative to repo root)")
    parser.add_argument("--n", type=int, default=None, help="Number of samples")
    parser.add_argument("--mode", choices=["index", "insert"], default="index",
                        help="'index' overwrites, 'insert' appends")
    parser.add_argument("--doc-collection", default="baseline_document_data")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters (default: 500)")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap in characters (default: 50)")
    args = parser.parse_args()

    t_start = time.time()

    # ── Banner ──────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold white]Baseline Indexing[/bold white]\n"
        "[dim]Simple fixed-size chunking • No LLM • Document-only RAG[/dim]",
        border_style="cyan", padding=(1, 3),
    ))
    console.print()

    # ── Step 1: Load dataset ────────────────────────────
    console.print(Rule("[bold cyan]Step 1/5[/bold cyan]  Load Dataset", style="cyan"))
    dataset_path = ROOT / args.dataset
    df = pd.read_csv(dataset_path)
    if args.n:
        df = df.head(args.n)

    para_col = next((c for c in ["paragraph", "paragraphs"] if c in df.columns), None)
    if not para_col:
        raise ValueError(f"No paragraph column. Available: {df.columns.tolist()}")

    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_column(style="dim")
    info.add_column(style="bold white")
    info.add_row("Dataset", str(dataset_path.name))
    info.add_row("Samples", str(len(df)))
    info.add_row("Mode", args.mode)
    info.add_row("Collection", args.doc_collection)
    info.add_row("Chunk size", f"{args.chunk_size} chars  (overlap {args.chunk_overlap})")
    console.print(info)
    console.print()

    # ── Step 2: Chunk documents ─────────────────────────
    console.print(Rule("[bold cyan]Step 2/5[/bold cyan]  Chunk Documents", style="cyan"))
    all_chunks = []
    with _progress_bar() as progress:
        task = progress.add_task("Chunking", total=len(df))
        for _, row in df.iterrows():
            doc_text = parse_paragraph(row[para_col])
            chunks = fixed_size_chunk(doc_text, args.chunk_size, args.chunk_overlap)
            all_chunks.extend(chunks)
            progress.advance(task)
    console.print(f"  [green]✓[/green] {len(all_chunks)} chunks from {len(df)} documents")
    console.print()

    # ── Step 3: Init components ─────────────────────────
    console.print(Rule("[bold cyan]Step 3/5[/bold cyan]  Initialize Components", style="cyan"))
    config_path = ROOT / "src/urasys/config/models_config.json"
    with open(config_path) as f:
        models_config = ModelsConfig.from_dict(json.load(f))

    with console.status("[bold cyan]Connecting to embedder & Milvus…"):
        embedder = OpenAIEmbedder(config=OpenAIClientConfig(
            api_key=SETTINGS.OPENAI_API_KEY,
            model=models_config.embedding_config.model_id,
        ))
        vector_db = MilvusVectorDatabase(config=get_milvus_config(SETTINGS, run_async=False))

    if args.mode == "index":
        vector_db.create_collection(
            collection_name=args.doc_collection,
            collection_structure=DOCUMENT_DATABASE_SCHEMA,
        )
    console.print(f"  [green]✓[/green] Embedder: [bold]{models_config.embedding_config.model_id}[/bold]")
    console.print(f"  [green]✓[/green] Collection: [bold]{args.doc_collection}[/bold]  (mode={args.mode})")
    console.print()

    # ── Step 4: BM25 sparse vectors ─────────────────────
    console.print(Rule("[bold cyan]Step 4/5[/bold cyan]  Fit BM25 Sparse Vectors", style="cyan"))
    bm25_doc_path = ROOT / "src/urasys/data/bm25/baseline_document/state_dict.json"
    bm25_doc_path.parent.mkdir(parents=True, exist_ok=True)

    with console.status("[bold cyan]Fitting BM25 on chunks…"):
        doc_bm25 = BM25Client(language="en", init_without_load=True)
        chunk_sparse = doc_bm25.fit_transform(
            all_chunks, path=str(bm25_doc_path), auto_save_local=True
        )
    console.print(f"  [green]✓[/green] BM25 fitted on {len(all_chunks)} chunks")
    console.print(f"  [green]✓[/green] State dict → [dim]{bm25_doc_path}[/dim]")
    console.print()

    # ── Step 5: Embed + index ───────────────────────────
    console.print(Rule("[bold cyan]Step 5/5[/bold cyan]  Embed & Index into Milvus", style="cyan"))
    with _progress_bar() as progress:
        task = progress.add_task("Indexing", total=len(all_chunks))
        for idx, chunk in enumerate(all_chunks):
            dense = embedder.get_text_embedding(chunk)
            vector_db.insert_vectors(
                collection_name=args.doc_collection,
                data={
                    "chunk_id": str(uuid.uuid4()),
                    "chunk": chunk,
                    "chunk_dense_embedding": dense,
                    "chunk_sparse_embedding": chunk_sparse[idx],
                },
            )
            progress.advance(task)
    console.print()

    # ── Summary ─────────────────────────────────────────
    elapsed = time.time() - t_start
    summary = Table(title="Baseline Indexing Complete", box=box.ROUNDED,
                    title_style="bold green", border_style="green")
    summary.add_column("Metric", style="dim")
    summary.add_column("Value", style="bold white")
    summary.add_row("Documents", str(len(df)))
    summary.add_row("Chunks indexed", str(len(all_chunks)))
    summary.add_row("Collection", args.doc_collection)
    summary.add_row("BM25 path", str(bm25_doc_path.relative_to(ROOT)))
    summary.add_row("Total time", f"{elapsed:.1f}s")
    console.print(summary)
    console.print()


if __name__ == "__main__":
    main()

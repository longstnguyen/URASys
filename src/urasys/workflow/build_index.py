from typing import List

import logging
import pandas as pd
from loguru import logger

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
)
from rich.rule import Rule

from urasys.core.model_clients import BM25Client, BaseEmbedder, BaseLLM
from urasys.indexing.context_document.base_class import PreprocessingConfig, ReconstructedChunk
from urasys.indexing.context_document import (
    ChunkReconstructor,
    ContextExtractor,
    SemanticChunker
)
from urasys.indexing.faq import (
    FaqAugmenter,
    FaqExpander,
    FaqGenerator
)
from urasys.indexing.faq.base_class import FAQDocument
from urasys.utils.base_class import IndexData
from urasys.utils.database_clients import BaseVectorDatabase
from urasys.utils.vectordb_schema import (
    DOCUMENT_DATABASE_SCHEMA,
    FAQ_DATABASE_SCHEMA,
    JSON_INDEX_PARAMS
)


# Turn off logging for OpenAI calls
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_console = Console()


def _progress_bar() -> Progress:
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="green", finished_style="bold green"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=_console,
    )


class DataIndex:
    def __init__(
        self,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        document_bm25_client: BM25Client,
        faq_bm25_client: BM25Client,
        preprocessing_config: PreprocessingConfig,
        vector_db: BaseVectorDatabase
    ):
        """
        Initialize the DataIndex class.
        This class is responsible for indexing documents and FAQ context into a vector database.

        Args:
            llm (BaseLLM): Language model used for context extraction.
            embedder (BaseEmbedder): Embedder model to embed the documents into dense vectors.
            document_bm25_client (BM25Client): BM25 client to embed the documents into sparse vectors.
            faq_bm25_client (BM25Client): BM25 client to embed the FAQs into sparse vectors.
            preprocessing_config (PreprocessingConfig): Configuration for preprocessing text.
            vector_db (BaseVectorDatabase): Vector database client for storing indexed data.
        """
        self.semantic_chunker = SemanticChunker(
            embedder=embedder,
            preprocessing_config=preprocessing_config,
            breakpoint_percentile_threshold=95,
            min_chunk_size=10,
            max_chunk_size=1000,
        )
        self.context_extractor = ContextExtractor(llm)
        self.chunk_reconstructor = ChunkReconstructor(llm)
        self.faq_expander = FaqExpander(llm)
        self.faq_generator = FaqGenerator(llm)
        self.faq_augmenter = FaqAugmenter(llm)
        self.vector_db = vector_db
        self.embedder = embedder
        self.document_bm25_client = document_bm25_client
        self.faq_bm25_client = faq_bm25_client

    def run_index(
        self,
        documents: List[str] = [],
        faqs: List[FAQDocument] = [],
        document_collection_name: str = "document_collection",
        faq_collection_name: str = "faq_collection"
    ) -> IndexData:
        """
        Run the indexing process for documents and FAQs.

        Args:
            documents (List[str]): List of documents to index.
            faqs (List[FAQDocument]): List of FAQ documents to index.
            document_collection_name (str): Name of the collection for documents.
            faq_collection_name (str): Name of the collection for FAQs.
        """
        # Build index
        index_data = self.build_index(
            documents=documents,
            faqs=faqs,
            document_collection_name=document_collection_name,
            faq_collection_name=faq_collection_name,
            overwrite_collection=True
        )

        # Index data
        self.index(
            index_data=index_data,
            document_collection_name=document_collection_name,
            faq_collection_name=faq_collection_name
        )

        return index_data
    
    def run_insert(
        self,
        documents: List[str] = [],
        faqs: List[FAQDocument] = [],
        document_collection_name: str = "document_collection",
        faq_collection_name: str = "faq_collection"
    ) -> IndexData:
        """
        Run the inserting process for documents and FAQs.

        Args:
            documents (List[str]): List of documents to insert.
            faqs (List[FAQDocument]): List of FAQ documents to insert.
            document_collection_name (str): Name of the collection for documents.
            faq_collection_name (str): Name of the collection for FAQs.
        """
        # Build index for new data
        index_data = self.build_index(
            documents=documents,
            faqs=faqs,
            document_collection_name=document_collection_name,
            faq_collection_name=faq_collection_name,
            overwrite_collection=False
        )

        # Insert the new index data into the vector database
        self.insert(
            index_data=index_data,
            document_collection_name=document_collection_name,
            faq_collection_name=faq_collection_name
        )

        return index_data

    def index(
        self,
        index_data: IndexData,
        document_collection_name: str = "document_collection",
        faq_collection_name: str = "faq_collection"
    ) -> None:
        """
        Index the given data into the vector database.
        """
        # Index documents
        if index_data.documents:
            _console.print(Rule("[bold blue]Indexing Documents[/bold blue]", style="blue"))

            # Fit the BM25 client to the documents
            with _console.status("[bold cyan]Fitting BM25 on document chunks…"):
                chunk_sparse_embeddings = self.document_bm25_client.fit_transform(
                    [document.chunk for document in index_data.documents],
                    path="./src/urasys/data/bm25/document/state_dict.json",
                    auto_save_local=True
                )
            _console.print(f"  [green]✓[/green] BM25 fitted on {len(index_data.documents)} chunks")

            with _progress_bar() as progress:
                task = progress.add_task("Embed + insert docs", total=len(index_data.documents))
                for idx, document in enumerate(index_data.documents):
                    dense_embedding = self.embedder.get_text_embedding(document.chunk)
                    self.vector_db.insert_vectors(
                        collection_name=document_collection_name,
                        data={
                            "chunk_id": document.id,
                            "chunk": document.chunk,
                            "chunk_dense_embedding": dense_embedding,
                            "chunk_sparse_embedding": chunk_sparse_embeddings[idx]
                        }
                    )
                    progress.advance(task)

        # Index FAQs
        if index_data.faqs:
            _console.print(Rule("[bold blue]Indexing FAQs[/bold blue]", style="blue"))

            with _console.status("[bold cyan]Fitting BM25 on FAQs…"):
                faq_sparse_embeddings = self.faq_bm25_client.fit_transform(
                    [(faq.question + " " + faq.answer) for faq in index_data.faqs],
                    path="./src/urasys/data/bm25/faq/state_dict.json",
                    auto_save_local=True
                )
            _console.print(f"  [green]✓[/green] BM25 fitted on {len(index_data.faqs)} FAQs")

            with _progress_bar() as progress:
                task = progress.add_task("Embed + insert FAQs", total=len(index_data.faqs))
                for idx, faq in enumerate(index_data.faqs):
                    combined_faq = faq.question + " " + faq.answer
                    dense_embedding = self.embedder.get_text_embedding(combined_faq)
                    self.vector_db.insert_vectors(
                        collection_name=faq_collection_name,
                        data={
                            "faq_id": faq.id,
                            "faq": {
                                "question": faq.question,
                                "answer": faq.answer
                            },
                            "question_dense_embedding": dense_embedding,
                            "question_sparse_embedding": faq_sparse_embeddings[idx]
                        }
                    )
                    progress.advance(task)

        _console.print(f"  [green]✓[/green] Indexing completed")

    def insert(
        self,
        index_data: IndexData,
        document_collection_name: str = "document_collection",
        faq_collection_name: str = "faq_collection"
    ) -> None:
        """
        Insert the new index data into the vector database, updating the old index data.

        Args:
            index_data (IndexData): New data to be inserted.
            document_collection_name (str): Name of the collection for documents.
            faq_collection_name (str): Name of the collection for FAQs.
        """

        logger.info("Inserting new index data and updating old index data...")

        # Get indexed dense embeddings
        chunk_data = self.vector_db.get_items(
            collection_name=document_collection_name,
            field_names=["chunk_id", "chunk", "chunk_dense_embedding"]
        )
        faq_data = self.vector_db.get_items(
            collection_name=faq_collection_name,
            field_names=["faq_id", "faq", "question_dense_embedding"]
        )
        
        # Store the dense embeddings in a dictionary
        chunk_embedding_dict = {
            chunk["chunk_id"]: chunk["chunk_dense_embedding"] for chunk in chunk_data
        }
        faq_embedding_dict = {
            faq["faq_id"]: faq["question_dense_embedding"] for faq in faq_data
        }

        # Add dense embeddings for the new data
        dense_embeddings = self.embedder.get_text_embeddings(
            [document.chunk for document in index_data.documents]
        )
        for idx, document in enumerate(index_data.documents):
            # Update the dense embedding in the dictionary
            chunk_embedding_dict[document.id] = dense_embeddings[idx]

        dense_embeddings = self.embedder.get_text_embeddings(
            [(faq.question + " " + faq.answer) for faq in index_data.faqs]
        )
        for idx, faq in enumerate(index_data.faqs):
            # Update the dense embedding in the dictionary
            faq_embedding_dict[faq.id] = dense_embeddings[idx]

        # Update BM25 state dicts
        total_chunks = [document.chunk for document in index_data.documents] + [data["chunk"] for data in chunk_data]
        chunk_sparse_embeddings = self.document_bm25_client.fit_transform(
            data=total_chunks,
            path="./src/urasys/data/bm25/document/state_dict.json",
            auto_save_local=True
        )
        total_faqs = [(faq.question + " " + faq.answer) for faq in index_data.faqs] + [data["faq"]["question"] for data in faq_data]
        faq_sparse_embeddings = self.faq_bm25_client.fit_transform(
            data=total_faqs,
            path="./src/urasys/data/bm25/faq/state_dict.json",
            auto_save_local=True
        )

        # Add old chunk data into index data
        index_data.documents.extend([
            ReconstructedChunk(
                id=data["chunk_id"],
                chunk= data["chunk"]
            )
            for data in chunk_data
        ])
        # Add old FAQ data into index data
        index_data.faqs.extend([
            FAQDocument(
                id=data["faq_id"],
                question=data["faq"]["question"],
                answer=data["faq"]["answer"]
            )
            for data in faq_data
        ])

        # Clear the old data in the vector database
        logger.info("Deleting old data in the vector database...")
        self._create_collection(document_collection_name, faq_collection_name)

        # Index all chunk data into the vector database
        for idx, document in enumerate(index_data.documents):
            # Get dense embedding from the dictionary
            dense_embedding = chunk_embedding_dict[document.id]

            # Index the document into the vector database
            self.vector_db.insert_vectors(
                collection_name=document_collection_name,
                data={
                    "chunk_id": document.id,
                    "chunk": document.chunk,
                    "chunk_dense_embedding": dense_embedding,
                    "chunk_sparse_embedding": chunk_sparse_embeddings[idx]
                }
            )

        # Index all FAQ data into the vector database
        for idx, faq in enumerate(index_data.faqs):
            # Get dense embedding from the dictionary
            dense_embedding = faq_embedding_dict[faq.id]

            # Index all FAQs into the vector database
            self.vector_db.insert_vectors(
                collection_name=faq_collection_name,
                data={
                    "faq_id": faq.id,
                    "faq": {
                        "question": faq.question,
                        "answer": faq.answer
                    },
                    "question_dense_embedding": dense_embedding,
                    "question_sparse_embedding": faq_sparse_embeddings[idx]
                }
            )

    def build_index(
        self,
        documents: List[str] = [],
        faqs: List[FAQDocument] = [],
        document_collection_name: str = "document_collection",
        faq_collection_name: str = "faq_collection",
        overwrite_collection: bool = True
    ) -> IndexData:
        """
        Build index for documents and FAQ context.
        """
        # ------ Create collections ------
        if overwrite_collection:
            self._create_collection(document_collection_name, faq_collection_name)

        # ------ Build document context ------
        _console.print(Rule("[bold blue]Build Document Context[/bold blue]", style="blue"))
        _console.print("  [dim]Extract context → Semantic chunk → Reconstruct chunks[/dim]")

        reconstructed_chunks = []

        with _progress_bar() as progress:
            task = progress.add_task("Processing docs", total=len(documents))
            for document in documents:
                # Step 1: Extract context from documents
                extracted_context = self.context_extractor.extract_context_documents([document])

                # Step 2: Chunk documents
                text_chunks = self.semantic_chunker.chunk([document])

                # Step 3: Reconstruct documents
                if not extracted_context:
                    logger.error("Skipping document: context extraction failed after all retries.")
                    progress.advance(task)
                    continue

                reconstructed_chunks.extend(self.chunk_reconstructor.reconstruct_chunks(
                    chunks=text_chunks,
                    context=extracted_context[0]
                ))
                progress.advance(task)

        _console.print(f"  [green]✓[/green] {len(reconstructed_chunks)} chunks from {len(documents)} documents")
        _console.print()

        # ------ Build FAQ context ------
        _console.print(Rule("[bold blue]Build FAQ Context[/bold blue]", style="blue"))
        _console.print(f"  [dim]Input: {len(faqs)} FAQs + {len(reconstructed_chunks)} chunks[/dim]")

        processed_faqs = faqs

        # Step 1: Expand FAQ pairs from existing ones
        if processed_faqs:
            with _console.status(f"[bold cyan]Expanding {len(processed_faqs)} FAQ pairs…"):
                expanded_faqs = self.faq_expander.expand_faq(processed_faqs)
                processed_faqs.extend(expanded_faqs)
            _console.print(f"  [green]✓[/green] Expanded: +{len(expanded_faqs)} → {len(processed_faqs)} FAQs")

        # Step 2: Generate new FAQ pairs from reconstructed chunks
        if reconstructed_chunks:
            with _console.status(f"[bold cyan]Generating FAQs from {len(reconstructed_chunks)} chunks…"):
                generated_faqs = self.faq_generator.generate_faq(reconstructed_chunks)
                processed_faqs.extend(generated_faqs)
            _console.print(f"  [green]✓[/green] Generated: +{len(generated_faqs)} → {len(processed_faqs)} FAQs")

        # Step 3: Enrich FAQ pairs
        if processed_faqs:
            with _console.status(f"[bold cyan]Augmenting {len(processed_faqs)} FAQs…"):
                enriched_faqs = self.faq_augmenter.augment_faq(processed_faqs)
                processed_faqs.extend(enriched_faqs)
            _console.print(f"  [green]✓[/green] Augmented: +{len(enriched_faqs)} → {len(processed_faqs)} FAQs")

        _console.print()

        return IndexData(
            documents=reconstructed_chunks,
            faqs=processed_faqs
        )

    def _create_collection(self, document_collection_name: str, faq_collection_name: str):
        """Create collections in the vector database."""
        with _console.status("[bold cyan]Creating Milvus collections…"):
            self.vector_db.create_collection(
                collection_name=document_collection_name,
                collection_structure=DOCUMENT_DATABASE_SCHEMA
            )
            self.vector_db.create_collection(
                collection_name=faq_collection_name,
                collection_structure=FAQ_DATABASE_SCHEMA,
                json_index_params=JSON_INDEX_PARAMS
            )
        _console.print(f"  [green]✓[/green] Collection: [bold]{document_collection_name}[/bold]")
        _console.print(f"  [green]✓[/green] Collection: [bold]{faq_collection_name}[/bold]")


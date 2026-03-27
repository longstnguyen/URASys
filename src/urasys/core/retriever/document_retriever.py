from typing import Dict, List

from urasys.core.retriever.base_class import (
    DocumentNode,
    RetrievedDocument,
    DocumentRetrievalResult
)
from urasys.core.retriever.base_retriever import BaseHybridRetriever
from urasys.core.model_clients import BM25Client, BaseEmbedder
from urasys.utils.database_clients import BaseVectorDatabase


class DocumentRetriever(BaseHybridRetriever):
    """
    Concrete implementation of a hybrid retriever for document retrieval.
    
    This class implements the methods to retrieve documents from a vector database
    using both dense and sparse embeddings.
    
    Example:
    >>> retriever = DocumentRetriever("collection_name", embedder, bm25_client, vector_db)
    >>> results = retriever.retrieve_documents("How to apply for admission?", top_k=3)
    """

    def __init__(
        self,
        collection_name: str,
        embedder: BaseEmbedder,
        bm25_client: BM25Client,
        vector_db: BaseVectorDatabase
    ):
        super().__init__(collection_name, embedder, bm25_client, vector_db)
        self.embedding_fields = {
            "dense": "chunk_dense_embedding",
            "sparse": "chunk_sparse_embedding"
        }
        self.output_fields = ["chunk_id", "chunk"]

    def retrieve_documents(self, query: str, top_k: int = 5) -> DocumentRetrievalResult:
        """
        Retrieve documents from the vector database based on the query.
        
        Args:
            query (str): The query string to search for.
            top_k (int): The number of top documents to retrieve. Defaults to 5.
        
        Returns:
            DocumentRetrievalResult: Result contains the query and a list of retrieved documents.
        """
        # Perform hybrid search
        results = self.retrieve(
            query=query,
            field_names=self.embedding_fields,
            output_fields=self.output_fields,
            top_k=top_k
        )

        retrieved_nodes = DocumentRetrievalResult(
            query=query,
            documents=[
                RetrievedDocument(
                    source_node=DocumentNode(
                        id=result["chunk_id"],
                        chunk=result["chunk"]
                    ),
                    score=result.get("_score", 0.0)
                ) for result in results
            ]
        )

        return retrieved_nodes
    
    def get_field_names(self) -> Dict[str, str]:
        """Get the field names for the embeddings used in the retrieval."""
        return self.embedding_fields
    
    def get_output_fields(self) -> List[str]:
        """Get the output fields for the retrieval results."""
        return self.output_fields


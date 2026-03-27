from typing import Dict, List

from urasys.core.retriever.base_class import (
    FAQNode,
    RetrievedFAQ,
    FAQRetrievalResult
)
from urasys.core.retriever.base_retriever import BaseHybridRetriever
from urasys.core.model_clients import BM25Client, BaseEmbedder
from urasys.utils.database_clients import BaseVectorDatabase


class FAQRetriever(BaseHybridRetriever):
    """
    Concrete implementation of a hybrid retriever for FAQ retrieval.
    
    This class implements the methods to retrieve FAQs from a vector database
    using both dense and sparse embeddings.
    
    Example:
    >>> retriever = DocumentRetriever("collection_name", embedder, bm25_client, vector_db)
    >>> results = retriever.retrieve_faqs("How to apply for admission?", top_k=3)
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
            "dense": "question_dense_embedding",
            "sparse": "question_sparse_embedding"
        }
        self.output_fields = ["faq_id", "faq"]

    def retrieve_faqs(self, query: str, top_k: int = 5) -> FAQRetrievalResult:
        """
        Retrieve FAQs from the vector database based on the query.
        
        Args:
            query (str): The query string to search for.
            top_k (int): The number of top FAQs to retrieve. Defaults to 5.
        
        Returns:
            FAQRetrievalResult: Result contains the query and a list of retrieved FAQs.
        """
        # Perform hybrid search
        results = self.retrieve(
            query=query,
            field_names=self.embedding_fields,
            output_fields=self.output_fields,
            top_k=top_k
        )

        retrieved_nodes = FAQRetrievalResult(
            query=query,
            faqs=[
                RetrievedFAQ(
                    source_node=FAQNode(
                        id=result["faq_id"],
                        question=result["faq"]["question"],
                        answer=result["faq"]["answer"],
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


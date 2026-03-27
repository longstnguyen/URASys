from typing import Dict, List
import logging

from urasys.core.model_clients import BM25Client, BaseEmbedder
from urasys.utils.database_clients import BaseVectorDatabase
from urasys.utils.database_clients.base_class import EmbeddingData, EmbeddingType
from urasys.utils.database_clients.milvus.utils import MetricType

# Turn off logging for OpenAI calls
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class BaseHybridRetriever:
    """
    Base class for retrieving documents using a hybrid search approach combining dense and sparse embeddings.
    
    This class serves as a foundation for implementing various retrieval strategies using vector databases.
    It supports hybrid search combining embedding-based semantic similarity (dense) with keyword-based 
    relevance (sparse/BM25) to achieve more accurate retrieval results.
    
    Subclasses should implement specific retrieval strategies for different data types,
    such as FAQ documents, web content, or context documents, by overriding methods as needed.
    
    Attributes:
        collection_name (str): Name of the collection in the vector database.
        embedder (BaseEmbedder): Model for generating dense embeddings.
        bm25_client (BM25Client): Client for generating sparse/BM25 embeddings.
        vector_db (BaseVectorDatabase): Vector database client for performing searches.
    
    Methods:
        retrieve: Retrieve relevant documents based on a query.
        get_field_names: Get the field names for dense and sparse embeddings (to be implemented by subclasses).
        get_output_fields: Get the fields to include in the output (to be implemented by subclasses).
        process_results: Process the retrieved results (to be implemented by subclasses).
    
    Example:
        >>> embedder = BaseEmbedder()
        >>> bm25_client = BM25Client()
        >>> vector_db = BaseVectorDatabase()
        >>> retriever = BaseHybridRetriever("collection_name", embedder, bm25_client, vector_db)
        >>> results = retriever.retrieve("How to apply for admission?", top_k=3)
    """
    def __init__(
        self,
        collection_name: str,
        embedder: BaseEmbedder,
        bm25_client: BM25Client,
        vector_db: BaseVectorDatabase
    ):
        self.collection_name = collection_name
        self.embedder = embedder
        self.bm25_client = bm25_client
        self.vector_db = vector_db

        # Initialize the vector database collection
        self.vector_db.load_collection(collection_name=self.collection_name)

    def retrieve(
        self,
        query: str,
        field_names: Dict[str, str],
        output_fields: List[str],
        top_k: int = 5
    ) -> List[dict]:
        """
        Retrieve documents from the vector database based on the query.
        
        Args:
            query (str): The query string to search for.
            field_names (Dict[str, str]): A dictionary mapping embedding types 
                (`"dense"` and `"sparse"`) to expected field names.
            output_fields (List[str]): The fields to include in the output.
            top_k (int): The number of top documents to retrieve. Defaults to 5.
        
        Returns:
            List[dict]: The output is a list of dictionaries containing the search results.
                Each dictionary corresponds to a search result and contains the fields specified 
                in `output_fields` and `_score` field.
        """
        # Embed the query
        query_dense_embedding = self.embedder.get_query_embedding(query)
        query_sparse_embedding = self.bm25_client.encode_queries([query])[0]

        # Prepare the embedding data
        dense_data = EmbeddingData(
            field_name=field_names["dense"],
            embeddings=query_dense_embedding,
            embedding_type=EmbeddingType.DENSE
        )
        sparse_data = EmbeddingData(
            field_name=field_names["sparse"],
            embeddings=query_sparse_embedding,
            embedding_type=EmbeddingType.SPARSE
        )

        # Perform hybrid search
        results = self.vector_db.hybrid_search_vectors(
            embedding_data=[dense_data, sparse_data],
            output_fields=output_fields,
            top_k=top_k,
            collection_name=self.collection_name,
            metric_type=MetricType.IP,  # Use Inner Product for dense embeddings
        )
        
        return results


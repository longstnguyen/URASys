from urasys.utils.database_clients.base_class import (
    VectorDBBackend,
    VectorDBConfig
)


class MilvusConfig(VectorDBConfig):
    """
    Configuration for Milvus database.

    Supports three connection modes (auto-detected from the ``uri`` that is built):

    1. **Milvus Lite** – pass ``local_db_path="./path/to/file.db"``
       No server required; data is stored in a local file via the embedded engine.
    2. **Self-hosted Milvus** – pass ``host`` and ``port``
       Connects to a Milvus server running on Docker / K8s.
    3. **Zilliz Cloud** – pass ``cloud_uri`` and ``token``
       Connects to a managed Zilliz Cloud instance.

    Priority: ``local_db_path`` > ``cloud_uri`` > ``host:port``.

    Args:
        host (str): Hostname of the Milvus server. Defaults to "localhost".
        port (str): Port number of the Milvus server. Defaults to "19530".
        cloud_uri (str, optional): Cloud URI for Milvus. If provided, overrides host and port.
        token (str, optional): Authentication token for Milvus cloud connection.
        local_db_path (str, optional): File path for Milvus Lite embedded mode (e.g. "./milvus.db").
        run_async (bool): Whether to run the client asynchronously. Defaults to False.
    """
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        cloud_uri: str = "",
        token: str = "",
        local_db_path: str = "",
        run_async: bool = False,
        **kwargs
    ):
        super().__init__(VectorDBBackend.MILVUS, **kwargs)
        self.host = host
        self.port = port
        self.run_async = run_async
        self.uri = local_db_path or cloud_uri or f"http://{host}:{port}"
        self.token = token

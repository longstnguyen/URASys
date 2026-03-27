import pathlib

from urasys.utils.database_clients.milvus import MilvusConfig


def check_bool(text: str) -> bool:
    if text.lower() == "true":
        return True
    elif text.lower() == "false":
        return False
    else:
        raise ValueError(f"Invalid boolean value: {text}")


# Project root: <repo>/src/urasys/config/utils.py → <repo>
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_LOCAL_DB = str(_PROJECT_ROOT / "src" / "urasys" / "data" / "milvus" / "urasys.db")


def get_milvus_config(settings, run_async: bool = False) -> MilvusConfig:
    """Return a MilvusConfig based on ``STORAGE_MODE``.

    * ``"local"``  → Milvus Lite (file-based, no server needed)
    * ``"cloud"``  → Zilliz Cloud (requires ``MILVUS_CLOUD_URI`` / ``MILVUS_CLOUD_TOKEN``)
    """
    mode = settings.STORAGE_MODE.lower()
    if mode == "local":
        db_path = settings.MILVUS_LOCAL_DB_PATH or _DEFAULT_LOCAL_DB
        # Ensure the directory exists
        pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return MilvusConfig(local_db_path=db_path, run_async=run_async)
    elif mode == "cloud":
        return MilvusConfig(
            cloud_uri=settings.MILVUS_CLOUD_URI,
            token=settings.MILVUS_CLOUD_TOKEN,
            run_async=run_async,
        )
    else:
        raise ValueError(
            f"Unknown STORAGE_MODE '{settings.STORAGE_MODE}'. Use 'local' or 'cloud'."
        )

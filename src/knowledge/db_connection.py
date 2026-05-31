import os
import chromadb
from knowledge.chroma_config import get_chroma_db_path, get_collection_name


def get_db_collection():
    """Connect to ChromaDB using CHROMA_DB_PATH / CHROMA_COLLECTION from .env."""
    db_path = get_chroma_db_path()
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=get_collection_name())
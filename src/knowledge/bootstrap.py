"""
Ensure ChromaDB is populated before serving queries.

Local dev: uses ./medlabel_db (run ingest_data.py once).
Cloud: set MEDLABEL_DB_TAR_URL in Streamlit secrets to download a tarball,
       or commit medlabel_db/ to the deploy branch.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import urllib.request

from knowledge.chroma_config import get_chroma_db_path, get_collection_name


def _db_has_chunks() -> int:
    from knowledge.embedder import DrugEmbedder

    return DrugEmbedder().collection.count()


def _download_db_tarball(url: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        print(f"Downloading label DB from {url} ...")
        urllib.request.urlretrieve(url, tmp_path)
        with tarfile.open(tmp_path, "r:gz") as archive:
            archive.extractall(path=os.path.dirname(dest_dir))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def ensure_chroma_ready() -> int:
    """Return chunk count; download or ingest if the DB is empty."""
    db_path = get_chroma_db_path()
    coll_name = get_collection_name()

    try:
        count = _db_has_chunks()
        if count > 0:
            return count
    except Exception:
        pass

    tar_url = os.getenv("MEDLABEL_DB_TAR_URL", "").strip()
    if tar_url:
        parent = os.path.dirname(db_path)
        if os.path.isdir(db_path):
            shutil.rmtree(db_path)
        _download_db_tarball(tar_url, db_path)
        try:
            return _db_has_chunks()
        except Exception as exc:
            raise RuntimeError(
                f"Downloaded DB from MEDLABEL_DB_TAR_URL but could not open "
                f"{coll_name} at {db_path}: {exc}"
            ) from exc

    if not os.path.isdir(db_path) or not os.listdir(db_path):
        from knowledge.ingest_data import ingest_json_to_chroma

        print(
            "medlabel_db is empty — ingesting data/drugs.json "
            "(first cloud boot can take 10–20 min on CPU)..."
        )
        return ingest_json_to_chroma(reset=False)

    raise RuntimeError(
        f"Label database at {db_path} exists but collection "
        f"'{coll_name}' is empty. Run ingest locally or set MEDLABEL_DB_TAR_URL."
    )

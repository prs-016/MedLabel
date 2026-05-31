from __future__ import annotations

import json
import logging
from typing import Optional

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)


class BGEM3EmbeddingFunction(EmbeddingFunction):
    """
    Wraps BAAI/bge-m3 for ChromaDB.
    Tries FlagEmbedding first, falls back to
    sentence-transformers if not installed.
    """

    MODEL_NAME = "BAAI/bge-m3"

    def __init__(self):
        self._model   = None
        self._backend = "none"
        self._load()

    def _load(self):
        try:
            from FlagEmbedding import BGEM3FlagModel
            logger.info("Loading BGE-M3 via FlagEmbedding...")
            self._model   = BGEM3FlagModel(
                self.MODEL_NAME, use_fp16=True
            )
            self._backend = "flag"
            logger.info("BGE-M3 (FlagEmbedding) ready.")
            return
        except ImportError:
            logger.warning(
                "FlagEmbedding not installed. "
                "pip install FlagEmbedding"
            )
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(
                "Loading BGE-M3 via sentence-transformers..."
            )
            self._model   = SentenceTransformer(self.MODEL_NAME)
            self._backend = "st"
            logger.info("BGE-M3 (sentence-transformers) ready.")
        except Exception as exc:
            logger.error("Could not load BGE-M3: %s", exc)

    def __call__(self, input: Documents) -> Embeddings:
        if self._model is None:
            raise RuntimeError(
                "BGE-M3 not loaded. "
                "pip install FlagEmbedding"
            )
        if self._backend == "flag":
            result = self._model.encode(
                list(input),
                batch_size          = 12,
                max_length          = 8192,
                return_dense        = True,
                return_sparse       = False,
                return_colbert_vecs = False,
            )
            return result["dense_vecs"].tolist()
        else:
            vecs = self._model.encode(
                list(input), normalize_embeddings=True
            )
            return vecs.tolist()


class DrugEmbedder:

    def __init__(
        self,
        db_path:    str = "./medlabel_db",
        model_name: str = "BAAI/bge-m3",
    ):
        self.db_path    = db_path
        self.model_name = model_name
        self.client     = chromadb.PersistentClient(path=db_path)
        self._ef        = BGEM3EmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name               = "fda_medication_data",
        )
        logger.info(
            "ChromaDB ready at %s — %d chunks  [BGE-M3]",
            db_path, self.collection.count()
        )

    def upsert_chunks(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0
        ids, documents, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            if "id" in chunk:
                doc_id = chunk["id"]
            else:
                drug   = chunk["metadata"].get("drug_name", "unknown")
                sec    = chunk["metadata"].get("section_type", "unknown")
                source = chunk["metadata"].get("source", "unknown")
                doc_id = f"{drug}__{sec}__{source}__{i}"
            ids.append(doc_id)
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
        self.collection.upsert(
            ids=ids, documents=documents, metadatas=metadatas
        )
        logger.info(
            "Upserted %d chunks (first drug: '%s')",
            len(chunks),
            chunks[0]["metadata"].get("drug_name", "?"),
        )
        return len(chunks)

    def query(
        self,
        query_text: str,
        n_results:  int            = 5,
        where:      Optional[dict] = None,
    ) -> list[dict]:
        # Embed the query ourselves so the 1024-dim BGE-M3 vectors are used
        query_vec = self._ef([query_text])
        kwargs: dict = {
            "query_embeddings": query_vec,
            "n_results":        n_results,
        }
        if where:
            kwargs["where"] = where
        results = self.collection.query(**kwargs)
        return [
            {
                "text":     doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i, doc in enumerate(results["documents"][0])
        ]

    def export_to_json(
        self, output_path: str = "drug_db_export.json"
    ) -> int:
        data = self.collection.get(
            include=["documents", "metadatas"]
        )
        records = [
            {
                "id":       data["ids"][i],
                "text":     data["documents"][i],
                "metadata": data["metadatas"][i],
            }
            for i in range(len(data["ids"]))
        ]
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)
        logger.info(
            "Exported %d records → %s", len(records), output_path
        )
        return len(records)
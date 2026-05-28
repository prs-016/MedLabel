"""
BGE-M3 dense embeddings for ChromaDB retrieval.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"


class BGEEmbedder:
    """Lazy-loaded BGE-M3 via sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s …", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        vec = model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.tolist()

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
        )
        return [v.tolist() for v in vecs]

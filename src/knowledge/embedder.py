import logging
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class DrugEmbedder:

    def __init__(self, db_path: str = "./drug_db", model_name: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=db_path)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name)
        self.collection = self.client.get_or_create_collection(
            name="drugs",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB ready at %s — %d chunks", db_path, self.collection.count())

    def upsert_chunks(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0

        ids, documents, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            # DDInter docs carry a stable "id" field — use it to avoid collisions.
            # FDA/RxNorm chunks don't have one, so we build the old key.
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

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info(
            "Upserted %d chunks (first drug: '%s')",
            len(chunks),
            chunks[0]["metadata"].get("drug_name", "?"),
        )
        return len(chunks)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        kwargs = {"query_texts": [query_text], "n_results": n_results}
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

    def export_to_json(self, output_path: str = "drug_db_export.json") -> int:
        """
        Export the entire ChromaDB collection to a JSON file.
        Returns the number of records exported.
        """
        import json

        data = self.collection.get(include=["documents", "metadatas"])
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

        logger.info("Exported %d records → %s", len(records), output_path)
        return len(records)
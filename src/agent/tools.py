import sys
import os

# 1. Dynamically find the root 'MedLabel' folder and add it to Python's radar
# (This looks two folders up from src/agent/tools.py)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)

from src.knowledge.db_connection import get_db_collection
from sentence_transformers import CrossEncoder, SentenceTransformer

class MedReranker:
    def __init__(self, model_name=None):
        # Dynamically build the absolute path to your models folder
        if model_name is None:
            model_name = os.path.join(root_dir, "models", "medlabel_reranker")
            
        print(f"Loading Custom CrossEncoder from: {model_name}...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, retrieved_chunks: list[str], top_k: int = 3):
        """Reranks retrieved candidate chunks based on semantic alignment."""
        pairs = [[query, chunk] for chunk in retrieved_chunks]
        scores = self.model.predict(pairs)
        
        # Sort chunks by score in descending order
        scored_chunks = sorted(zip(scores, retrieved_chunks), key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k]]

def interaction_check(user_query: str):
    """
    Triggered by queries like 'Can I take warfarin with amoxicillin?'
    Searches only interaction-specific metadata in ChromaDB.
    """
    collection = get_db_collection()
    
    # We still need BGE-M3 for the initial fast retrieval
    embed_model = SentenceTransformer('BAAI/bge-m3')
    query_vector = embed_model.encode(user_query).tolist()
    
    print("\n🔍 Stage 1: Initial Retrieval (BGE-M3)")
    # We fetch 15 results, filtering specifically for interaction data
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=15,
        where={
            "$or": [
                {"section_type": "drug_interactions"},
                {"source": "ddinter2_web"}
            ]
        }
    )
    
    retrieved_chunks = results['documents'][0]
    
    if not retrieved_chunks:
        return "No interaction data found for these medications."

    unique_chunks = list(dict.fromkeys(retrieved_chunks))
    
    print(f"✅ Retrieved {len(unique_chunks)} unique broad matches. Passing to Cross-Encoder...")
    
    # Stage 2: Rerank the unique chunks down to the top 3
    reranker = MedReranker()
    best_chunks = reranker.rerank(query=user_query, retrieved_chunks=unique_chunks, top_k=3)
    
    print("\n🏆 Top Reranked Interaction Data:")
    for i, chunk in enumerate(best_chunks):
        print(f"\n--- Hit {i+1} ---")
        print(chunk)
        
    return best_chunks

if __name__ == "__main__":
    # Test the function
    interaction_check("Can I take warfarin safely with amoxicillin?")
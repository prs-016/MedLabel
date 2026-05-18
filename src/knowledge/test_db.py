from sentence_transformers import SentenceTransformer
from db_connection import get_db_collection

def verify_database():
    print("Connecting to ChromaDB...")
    # Grab the active collection directly from your new connection file
    collection = get_db_collection()

    # 1. The Count Check
    total_chunks = collection.count()
    print(f"\n📊 Total chunks in database: {total_chunks}")
    
    if total_chunks == 0:
        print("⚠️ The database is empty! Something went wrong with the ingestion.")
        return

    # 2. The Semantic Search Test
    print("\nLoading BGE-M3 model for a test query...")
    model = SentenceTransformer('BAAI/bge-m3')

    test_query = "What medication is used as a blood thinner to prevent blood clots?"
    print(f"\n🔍 Searching for: '{test_query}'")
    
    # Encode the question into a vector
    query_vector = model.encode(test_query).tolist()

    # Search the database for the top 3 closest matches
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3 
    )

    # 3. Print the Results
    print("\n--- Top 3 Matches ---")
    for i in range(len(results['documents'][0])):
        text_snippet = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        
        print(f"\nMatch {i+1} | Similarity Score: {distance:.4f}")
        print(f"💊 Drug: {metadata.get('drug_name').upper()}")
        print(f"📑 Section: {metadata.get('section_type')}")
        print(f"📝 Snippet: {text_snippet[:200]}...")

if __name__ == "__main__":
    verify_database()
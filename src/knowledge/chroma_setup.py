import chromadb
import os

def setup_database():
    """Initializes the ChromaDB client and creates the FDA data collection."""
    
    # 1. Get the exact path of the folder this script lives in (src/knowledge)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Build a bulletproof absolute path to MedLabel/data/chroma_db_storage
    db_path = os.path.join(script_dir, "..", "..", "data", "chroma_db_storage")
    
    # 3. Security check: Create the data folder if it doesn't exist yet!
    os.makedirs(db_path, exist_ok=True)
    
    # 4. Initialize the database
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="fda_medication_data")
    
    print(f"✅ ChromaDB successfully created at:\n{os.path.abspath(db_path)}")
    return collection

if __name__ == "__main__":
    setup_database()
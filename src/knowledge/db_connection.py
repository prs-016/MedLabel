import os
import chromadb
from dotenv import load_dotenv

# Force Python to load the .env file from the root of your project
# (It looks up two directories from src/knowledge/ to find the MedLabel root)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(root_dir, ".env"))

def get_db_collection():
    """Reads the .env file, connects to ChromaDB, and returns the collection."""
    
    # 1. Grab the path from the .env file (default to a fallback if it fails)
    db_path_env = os.getenv("CHROMA_DB_PATH", "./medlabel_db")
    
    # 2. Build the absolute path so it safely runs from anywhere
    absolute_db_path = os.path.join(root_dir, db_path_env.strip("./"))
    
    # 3. Create the folder if it doesn't exist
    os.makedirs(absolute_db_path, exist_ok=True)
    
    # 4. Connect and return!
    client = chromadb.PersistentClient(path=absolute_db_path)
    collection = client.get_or_create_collection(name="fda_medication_data")
    
    return collection
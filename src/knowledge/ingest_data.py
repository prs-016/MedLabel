import os
import json
import uuid
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from db_connection import get_db_collection

# --- Configuration & Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "drugs.json")

# The exact 50 drugs targeted for the database
FIFTY_DRUGS = [
    "warfarin", "metformin", "lisinopril", "atorvastatin", "amoxicillin",
    "metoprolol", "omeprazole", "amlodipine", "sertraline", "levothyroxine",
    "clopidogrel", "furosemide", "gabapentin", "hydrochlorothiazide", "fluoxetine",
    "alprazolam", "prednisone", "ciprofloxacin", "azithromycin", "losartan",
    "escitalopram", "tramadol", "clonazepam", "simvastatin", "pantoprazole",
    "carvedilol", "montelukast", "tamsulosin", "rosuvastatin", "duloxetine",
    "quetiapine", "albuterol", "bupropion", "spironolactone", "venlafaxine",
    "zolpidem", "trazodone", "meloxicam", "naproxen", "ibuprofen",
    "celecoxib", "rivaroxaban", "apixaban", "methotrexate", "cyclosporine",
    "tacrolimus", "digoxin", "phenytoin", "carbamazepine", "lithium"
]

def chunk_text(text, chunk_size=200, overlap=50):
    """Splits a long string into smaller chunks with overlapping words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def ingest_json_to_chroma():
    print("Connecting to ChromaDB...")
    collection = get_db_collection()

    print("Loading BGE-M3 model...")
    model = SentenceTransformer('BAAI/bge-m3')

    print(f"Loading data from {JSON_PATH}...")
    with open(JSON_PATH, 'r', encoding='utf-8') as file:
        all_drugs_data = json.load(file)
    
    all_ids = []
    all_documents = []
    all_embeddings = []
    all_metadata = []

    print("Filtering and processing the target drugs...")
    
    for item in tqdm(all_drugs_data, desc="Processing JSON"):
        
        # 🚨 THE FIX: Looking for "text" instead of "document"
        full_text = item.get("text", "") 
        meta = item.get("metadata", {})
        
        # Safely grab the drug name for matching
        raw_drug_name = meta.get("drug_name", "").lower()
        matched_drug = next((target for target in FIFTY_DRUGS if target in raw_drug_name), None)
        
        if not matched_drug or not full_text:
            continue

        text_chunks = chunk_text(full_text)
        
        # Extract valuable metadata for future RAG querying
        section_type = meta.get("section_type", "unknown_section")
        data_source = meta.get("source", "unknown_source")

        for chunk_index, chunk in enumerate(text_chunks):
            # Generate unique suffix to prevent ChromaDB collisions
            unique_suffix = uuid.uuid4().hex[:6]
            chunk_id = f"{matched_drug}_{section_type}_{chunk_index}_{unique_suffix}".replace(" ", "_")
            
            vector = model.encode(chunk).tolist()

            all_ids.append(chunk_id)
            all_documents.append(chunk)
            all_embeddings.append(vector)
            
            all_metadata.append({
                "drug_name": matched_drug,
                "section_type": section_type,
                "source": data_source,  # Now tracking where the info came from!
                "chunk_index": chunk_index
            })

    if not all_ids:
        print("\n⚠️ STOP: 0 chunks were created. Verify the JSON structure.")
        return

    print(f"\nSaving {len(all_ids)} vectors to database in batches...")
    
    # Set a batch size safely below the 5461 limit
    batch_size = 5000 
    
    # Loop through our massive lists and insert them in chunks of 5000
    for i in range(0, len(all_ids), batch_size):
        end_index = min(i + batch_size, len(all_ids))
        print(f"➡️ Inserting batch {i} to {end_index}...")
        
        collection.add(
            ids=all_ids[i:end_index],
            embeddings=all_embeddings[i:end_index],
            documents=all_documents[i:end_index],
            metadatas=all_metadata[i:end_index]
        )
    
    print("✅ Ingestion complete! Database is ready.")

if __name__ == "__main__":
    ingest_json_to_chroma()
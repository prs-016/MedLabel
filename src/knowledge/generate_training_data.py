import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from db_connection import get_db_collection

# Force Python to load the .env file from the root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(root_dir, ".env"))

# Initialize the Grok client using the standard OpenAI library
client = OpenAI(
    api_key=os.environ.get("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

def generate_grok_prompts():
    print("Connecting to ChromaDB...")
    collection = get_db_collection()
    
    # Let's pull 50 chunks to start. (You can increase this to 500+ later for real training)
    print("Fetching 50 chunks for training data generation...")
    db_data = collection.get(limit=50)
    chunks = db_data['documents']
    
    training_pairs = []

    print(f"Generating synthetic queries using Grok...\n")
    
    for i, chunk in enumerate(chunks):
        # We wrap this in a try-except block just in case the API times out
        try:
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # The exact prompt engineering instructions for Grok
            system_prompt = (
                "You are an expert medical data annotator generating training data for a Cross-Encoder RAG system. "
                "You must respond ONLY with a raw JSON object containing exactly two keys: 'good_query' and 'bad_query'. "
                "Do not include markdown blocks like ```json."
            )
            
            user_prompt = f"""
            Read the following medical text chunk:
            "{chunk}"
            
            1. 'good_query': Write one highly relevant user question that this text perfectly and explicitly answers.
            2. 'bad_query': Write one trick question that mentions similar medical keywords from the text, but the text DOES NOT actually answer it.
            """

            response = client.chat.completions.create(
                model="grok-4.3", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2 # Keep it low so Grok stays strictly factual
            )

            # Parse the JSON response from Grok
            grok_output = json.loads(response.choices[0].message.content.strip())
            
            # Save the Good Pair (Label 1.0)
            training_pairs.append({
                "query": grok_output["good_query"],
                "chunk": chunk,
                "label": 1.0
            })
            
            # Save the Bad Pair (Label 0.0)
            training_pairs.append({
                "query": grok_output["bad_query"],
                "chunk": chunk,
                "label": 0.0
            })
            
            # Brief pause to respect API rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error on chunk {i+1}: {e}")
            continue

    # Save the generated pairs to your data folder
    output_path = os.path.join(root_dir, "data", "training_pairs.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_pairs, f, indent=4)
        
    print(f"\n✅ Successfully generated {len(training_pairs)} training pairs!")
    print(f"Data saved to: {output_path}")

if __name__ == "__main__":
    generate_grok_prompts()
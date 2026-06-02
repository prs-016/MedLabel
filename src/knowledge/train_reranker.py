import os
import json
import torch
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample

# --- Setup Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Interaction-specific pairs (generate_interaction_data.py). Falls back to the
# legacy generic training_pairs.json only if the new file is absent.
DATA_PATH = os.path.join(ROOT_DIR, "data", "interaction_pairs.json")
if not os.path.isfile(DATA_PATH):
    DATA_PATH = os.path.join(ROOT_DIR, "data", "training_pairs.json")
MODEL_OUTPUT_PATH = os.path.join(ROOT_DIR, "models", "medlabel_reranker")

def train_custom_reranker():
    print(f"Loading training data from {DATA_PATH}...")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    train_samples = []
    for item in raw_data:
        query = item["query"]
        chunk = item["chunk"]
        label = float(item["label"]) 
        train_samples.append(InputExample(texts=[query, chunk], label=label))

    print(f"Loaded {len(train_samples)} training pairs.")

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

    print("\nInitializing the Cross-Encoder model...")
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model.model.to(device)
    print(f"Training on device: {device.upper()}")

    # 🚨 FIX 1: Ensure the directory actually exists before we try to save to it
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

    use_amp = device == "cuda"

    print("\nStarting training (this might take a few minutes)...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=3,           
        warmup_steps=10,    
        output_path=MODEL_OUTPUT_PATH,
        use_amp=use_amp,
    )

    # 🚨 FIX 2: Explicitly force the model to save its weights to the hard drive!
    print("\nWriting model files to disk...")
    model.save(MODEL_OUTPUT_PATH)

    print(f"\n✅ Training complete! Custom MedLabel model physically saved to:\n{MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train_custom_reranker()
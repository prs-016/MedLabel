"""
src/knowledge/train_vector_search.py
=====================================
Fine-tune a dedicated cross-encoder reranker for the *vector_search*
function and save it to models/vector_search_reranker/.

Input : data/vector_search_pairs.json   (query / chunk / label)
Output: models/vector_search_reranker/
"""

import json
import os

import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

ROOT_DIR          = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH         = os.path.join(ROOT_DIR, "data", "vector_search_pairs.json")
MODEL_OUTPUT_PATH = os.path.join(ROOT_DIR, "models", "vector_search_reranker")
BASE_MODEL        = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def train_vector_search_reranker() -> None:
    print(f"Loading training data from {DATA_PATH} ...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    train_samples = [
        InputExample(texts=[item["query"], item["chunk"]], label=float(item["label"]))
        for item in raw_data
    ]
    print(f"Loaded {len(train_samples)} training pairs.")

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

    print("\nInitializing CrossEncoder from base:", BASE_MODEL)
    model = CrossEncoder(BASE_MODEL, num_labels=1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model.model.to(device)
    print(f"Training on device: {device.upper()}")

    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

    # fp16 (use_amp) works on CUDA only; MPS raises in recent accelerate.
    use_amp = device == "cuda"

    print("\nStarting training (a few minutes)...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=3,
        warmup_steps=10,
        output_path=MODEL_OUTPUT_PATH,
        use_amp=use_amp,
    )

    print("\nWriting model files to disk...")
    model.save(MODEL_OUTPUT_PATH)
    print(f"\n✅ Training complete! vector_search reranker saved to:\n{MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train_vector_search_reranker()

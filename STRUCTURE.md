# MedLabel — Repository Structure

> This file maps every folder and file in the repo to its purpose and the team member responsible.

```
MedLabel/
│
├── README.md                       # Full project specification
├── STRUCTURE.md                    # This file — repo map
├── requirements.txt                # Python dependencies (create in Week 1)
├── .env.example                    # Template for API keys
├── .gitignore                      # Files excluded from git
│
├── data/                           # ALL DATA (not committed to git)
│   ├── raw/images/                 # Raw medicine bottle photos (Week 1)
│   │                               #   → Students 1 & 2 collect these
│   ├── labeled/                    # YOLO annotations (Week 1)
│   │                               #   → Output from Roboflow/LabelImg
│   ├── fda_labels/                 # Cached FDA label JSONs (Week 3)
│   │                               #   → Downloaded via scripts/download_fda_data.py
│   ├── ddinter/                    # DDInter interaction CSVs (Week 3)
│   │                               #   → Downloaded from ddinter2.scbdd.com
│   └── processed/                  # Chunked + processed text (Week 3)
│                                   #   → Output from ingestion pipeline
│
├── models/                         # TRAINED MODELS (not committed to git)
│   ├── yolo/                       # YOLOv11 weights after fine-tuning (Week 2)
│   │                               #   → best.pt, last.pt
│   └── embeddings/                 # Cached HuggingFace model files
│                                   #   → Auto-downloaded on first run
│
├── src/                            # CORE SOURCE CODE
│   ├── __init__.py
│   │
│   ├── vision/                     # Track A: Image → Text
│   │   ├── __init__.py
│   │   ├── detector.py             # YOLO label detection (Week 2)
│   │   │                           #   → Students 1 & 2
│   │   ├── preprocessor.py         # OpenCV image cleanup (Week 2)
│   │   │                           #   → Student 4
│   │   └── ocr.py                  # PaddleOCR wrapper (Week 2)
│   │                               #   → Student 3
│   │
│   ├── knowledge/                  # Track B: Data → Vector DB
│   │   ├── __init__.py
│   │   ├── ingest.py               # FDA data parsing + chunking (Week 3)
│   │   │                           #   → Student 1
│   │   ├── embedder.py             # Text → vector embedding (Week 3)
│   │   │                           #   → Student 2
│   │   └── vectorstore.py          # ChromaDB operations (Week 3)
│   │                               #   → Student 2
│   │
│   ├── agent/                      # Track C: Agentic Chatbot
│   │   ├── __init__.py
│   │   ├── brain.py                # LangChain agent setup (Week 4)
│   │   │                           #   → Students 1 & 2
│   │   ├── tools.py                # Agent tools: search, interactions (Week 4)
│   │   │                           #   → Students 1 & 2
│   │   ├── prompts.py              # System prompts + templates (Week 4)
│   │   │                           #   → Student 4
│   │   └── simplifier.py           # Text simplification logic (Week 4)
│   │                               #   → Student 4
│   │
│   └── api/                        # External API Clients
│       ├── __init__.py
│       ├── openfda.py              # openFDA API wrapper (Week 1)
│       │                           #   → Student 3
│       ├── rxnorm.py               # RxNorm name lookup (Week 1)
│       │                           #   → Student 3
│       └── ddinter.py              # DDInter data handler (Week 3)
│                                   #   → Student 3
│
├── app/                            # STREAMLIT WEB APP (Week 6)
│   ├── streamlit_app.py            # Main app entry point
│   ├── components/
│   │   ├── scanner.py              # Image upload + display component
│   │   ├── results.py              # Structured results display
│   │   └── chatbot.py              # Chat interface component
│   └── assets/
│       └── style.css               # Custom CSS styling
│
├── notebooks/                      # JUPYTER NOTEBOOKS (Exploration)
│   ├── 01_data_exploration.ipynb   # Explore FDA data structure
│   ├── 02_ocr_baseline.ipynb       # Test OCR on sample images
│   ├── 03_yolo_training.ipynb      # YOLO training and evaluation
│   ├── 04_rag_experiments.ipynb    # Test retrieval and prompts
│   └── 05_interaction_testing.ipynb # Test drug interaction queries
│
├── scripts/                        # UTILITY SCRIPTS
│   ├── download_fda_data.py        # Download FDA labels for starter set
│   ├── ingest_to_chromadb.py       # Populate ChromaDB from processed data
│   ├── train_yolo.py               # YOLO fine-tuning script
│   └── test_pipeline.py            # End-to-end smoke test
│
└── tests/                          # UNIT TESTS
    ├── test_ocr.py                 # Test OCR accuracy
    ├── test_vectorstore.py         # Test ChromaDB operations
    ├── test_agent.py               # Test agent tool routing
    └── test_interactions.py        # Test interaction lookup
```

## Who Owns What

| Student | Primary Modules | Weeks Active |
|---|---|---|
| **Student 1** | `vision/detector.py`, `knowledge/ingest.py`, `agent/brain.py` | 1–5 |
| **Student 2** | `vision/detector.py`, `knowledge/embedder.py`, `knowledge/vectorstore.py`, `agent/tools.py` | 1–5 |
| **Student 3** | `vision/ocr.py`, `api/openfda.py`, `api/rxnorm.py`, `api/ddinter.py` | 1–5 |
| **Student 4** | `vision/preprocessor.py`, `agent/prompts.py`, `agent/simplifier.py` | 1–5 |
| **All** | `app/` (Streamlit), `tests/`, presentation | 6–7 |

## Quick Start (For Students)

```bash
# 1. Clone the repo
git clone <repo-url>
cd MedLabel

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. Install dependencies (create requirements.txt first!)
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 5. Run the app (Week 6+)
streamlit run app/streamlit_app.py
```

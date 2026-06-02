<div align="center">

<img src="https://img.shields.io/badge/Live%20Demo-medlabel.streamlit.app-brightgreen?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo"/>
&nbsp;
<img src="https://img.shields.io/badge/FDA--Grounded-5%2C976%20chunks-blue?style=for-the-badge" alt="FDA Data"/>
&nbsp;
<img src="https://img.shields.io/badge/BGE--M3-Neural%20Search-purple?style=for-the-badge" alt="BGE-M3"/>
&nbsp;
<img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" alt="MIT License"/>

<br/><br/>

# MedLabel

### Point your phone at any medicine label. Get plain English answers instantly.

**[→ Try the live demo](https://medlabel.streamlit.app/)**

<br/>

</div>

---

## What is MedLabel?

MedLabel is a full-stack AI system that makes medication information accessible to everyone. Upload a photo of **any medicine packaging** — a pill bottle, cardboard box, blister strip, or prescription label — and MedLabel reads it, simplifies the instructions into plain English, and lets you ask follow-up questions like *"Can I take this with Advil?"* or *"What side effects should I watch for?"*

Every answer is grounded in **official FDA drug label data** — retrieved, ranked, and cited — never hallucinated.

---

## Features

| | Feature | Description |
|---|---|---|
| 📷 | **Smart Scanning** | YOLO geometry detection automatically routes each photo to the right OCR pipeline |
| 🔍 | **Neural Semantic Search** | BGE-M3 + Cross-Encoder reranking over 5,976 FDA label chunks |
| 💬 | **Agentic Chatbot** | Three specialized tools: dosage lookup, interaction check, adverse events |
| ⚠️ | **Interaction Checker** | Cross-references DDInter + FDA labels for drug-drug and drug-substance pairs |
| 📋 | **Plain-English Summary** | Medical jargon simplified to a 5th-grade reading level |
| 🔒 | **Zero Data Storage** | No user data, scans, or queries are retained |

---

## Architecture

```
Photo of Any Medicine Packaging
           │
           ▼
    ┌──────────────┐
    │  YOLO Router │  Classifies: flat label vs. cylindrical bottle
    └──────┬───────┘
           │
    ┌──────┴──────────┐
    │                 │
    ▼                 ▼
┌─────────┐    ┌──────────────┐
│ Path A  │    │   Path B     │
│ OpenCV  │    │   Vision     │  Both paths produce
│ + OCR   │    │  (curved)    │  identical structured output
└────┬────┘    └──────┬───────┘
     └────────┬───────┘
              │
              ▼
    ┌─────────────────────┐
    │  Plain-English      │
    │  Simplification     │
    └──────────┬──────────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
┌──────────┐    ┌─────────────────────┐
│ ChromaDB │    │  Agentic Chatbot    │
│ FDA Data │◀───│  ┌───────────────┐  │
│ DDInter  │    │  │ vector_search │  │
│ 5,976    │    │  │ interaction   │  │
│  chunks  │    │  │ adverse_events│  │
└──────────┘    │  └───────────────┘  │
                └─────────────────────┘
                          │
                          ▼
                   ┌────────────┐
                   │ Streamlit  │
                   │    UI      │
                   └────────────┘
```

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Geometry Router** | YOLOv11 (Roboflow) | Classifies flat vs. cylindrical packaging |
| **Flat OCR** | pytesseract / PaddleOCR | Extracts text from flat labels |
| **Curved Vision** | Vision API | Reads curved bottle text natively |
| **Embeddings** | BGE-M3 (BAAI/bge-m3) | 1024-dim neural embeddings; runs locally |
| **Reranking** | Cross-Encoder (sentence-transformers) | Precision reranking of retrieved chunks |
| **Vector Store** | ChromaDB | Local vector DB; FDA chunks + DDInter |
| **Drug Data** | openFDA Drug Label API | Official FDA labeling: dosage, warnings, interactions |
| **Interactions** | DDInter Database | Open-access drug-drug interaction data |
| **Adverse Events** | openFDA Adverse Events API | FDA-reported adverse event summaries |
| **Drug Normalization** | NIH RxNorm API | Brand name → Generic → RxCUI |
| **Web Interface** | Streamlit | Upload, scan, chat, summarize |

---

## The Chatbot — 3 Tools

The chatbot uses semantic search + a cross-encoder reranker to retrieve the most relevant FDA label chunks, then synthesizes a grounded answer.

```
vector_search       → Dosage, warnings, ingredients, general questions
interaction_check   → "Can I take this with Advil?" — DDInter + FDA cross-reference
adverse_events      → "What side effects have people reported?" — openFDA event data
```

A substance text-scan fallback handles non-drug queries like *"Can I drink alcohol with this?"* — scanning all label chunks for the substance when it isn't a named drug in the database.

---

## Data

- **5,976 FDA label chunks** across 57+ drugs — embedded with BGE-M3 and stored in ChromaDB
- **DDInter** drug-drug interaction pairs with severity + management recommendations
- **On-demand ingestion** — drugs not in the starter set are fetched from openFDA, chunked, embedded, and cached automatically on first query

Each chunk carries metadata for precise filtering:

```python
{
  "drug_name":    "ibuprofen",
  "brand_name":   "Advil",
  "rxcui":        "5640",
  "section_type": "warnings_and_cautions",
  "source":       "fda_label"
}
```

---

## Running Locally

**Prerequisites:** Python 3.10–3.12, tesseract binary

```bash
git clone https://github.com/prs-016/MedLabel.git
cd MedLabel

pip install -r requirements-full.txt   # full local stack (EasyOCR etc.)

# Set secrets
cp .env.example .env
# Add XAI_API_KEY, GEMINI_API_KEY, ROBOFLOW_API_KEY

streamlit run app/streamlit_app.py
```

**Cloud / lightweight install** (no PyTorch OCR models):

```bash
pip install -r requirements.txt
```

---

## Deployment

The live demo runs on **Streamlit Community Cloud** using the lightweight `requirements.txt` (no PyTorch, no EasyOCR — uses tesseract via `packages.txt`).

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medlabel.streamlit.app/)

---

## Safety

MedLabel is an informational tool — not a diagnostic or prescribing system.

- Answers use **only retrieved FDA data**. If the answer isn't in context, the chatbot refers to a pharmacist.
- Requests to exceed labeled doses are refused with the FDA warning.
- Low-confidence OCR reads are flagged visually.
- No user data or scans are stored.

> ⚠️ Always verify with the physical label or your pharmacist. This is not a substitute for professional medical advice.

---

## References

| Resource | Link |
|---|---|
| openFDA Drug Labels | https://open.fda.gov/apis/drug/label/ |
| openFDA Adverse Events | https://open.fda.gov/apis/drug/event/ |
| NIH RxNorm API | https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html |
| DDInter Database | [Kaggle](https://www.kaggle.com/datasets/thedevastator/ddinter-dataset-drug-drug-interactions) · [Zenodo](https://zenodo.org/records/5549420) |
| BGE-M3 | https://huggingface.co/BAAI/bge-m3 |
| ChromaDB | https://docs.trychroma.com/ |
| Roboflow | https://roboflow.com/ |

---

<div align="center">
  <sub>Built by the MedLabel team · UC San Diego</sub>
</div>

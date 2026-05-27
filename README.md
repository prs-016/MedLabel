# MedLabel — Intelligent Medicine Label Scanner & Agentic Chatbot

> **Difficulty:** Intermediate–Advanced · **Team Size:** 4 Students · **Duration:** 7 Weeks
> **Led by:** Prakhar

---

## What Is MedLabel?

MedLabel is a full-stack AI system that makes medication information accessible to everyone. Point your phone at **any medicine packaging** — a pill bottle, cardboard box, blister strip, cream tube, or prescription label — and MedLabel reads it, simplifies the instructions into plain English, and lets you chat with an AI assistant to ask follow-up questions like *"Can I take this with Advil?"* or *"What's the dose for my 8-year-old?"*

The AI assistant is powered by a **vector database of official FDA drug labels**, so every answer is grounded in real, cited medical data — never hallucinated.

---

## The Core Idea: A YOLO-Routed Hybrid Pipeline

Different medicine packaging has different geometry. MedLabel uses **YOLO as a router** to pick the right OCR tool for each format automatically:

| Packaging | Examples | Path | Tool |
|---|---|---|---|
| **Flat** | Cardboard boxes, blister strips, prescription labels, cream tubes | Path A | YOLO + OpenCV + PaddleOCR |
| **Cylindrical** | Pill bottles, syrup bottles, spray bottles | Path B | Gemini Vision AI |

Both paths produce identical structured output. Everything downstream — Gemini simplification, RxNorm normalization, ChromaDB retrieval, the LangChain chatbot — is shared.

---

## System Architecture

```
USER: Photo of Any Medicine Packaging
           │
           ▼
    ┌──────────────┐
    │  YOLOv11     │  Detects label region AND classifies geometry
    │  ROUTER      │  Class 0: flat_label / Class 1: cylindrical
    └──────┬───────┘
           │
    ┌──────┴──────┐
    │flat         │cylindrical
    ▼             ▼
┌────────┐   ┌──────────────┐
│OpenCV  │   │Gemini 2.0    │     Both paths output the same
│+Paddle │   │Flash (Vision)│ ──▶ structured JSON
│OCR     │   │              │
└────────┘   └──────────────┘
           │
           ▼
    ┌──────────────────────────────────────────┐
    │  GEMINI — Simplify to Plain English      │
    │  RxNorm — Brand Name → Generic → RxCUI  │
    └──────────────────┬───────────────────────┘
                       │
           ┌───────────┴────────────┐
           ▼                        ▼
    ┌─────────────┐       ┌──────────────────────┐
    │  ChromaDB   │       │  LangChain Agent     │
    │  Vector DB  │◀──────│  3 Tools:            │
    │  FDA Labels │       │  1. vector_search    │
    │  DDInter    │       │  2. interaction_check│
    │  User Scans │       │  3. adverse_events   │
    └─────────────┘       └──────────┬───────────┘
                                     ▼
                            ┌────────────────┐
                            │  Streamlit UI  │
                            └────────────────┘
```

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| **Packaging Router** | YOLOv11 (Ultralytics) | Detect label AND classify: flat vs. cylindrical |
| **Flat Preprocessing** | OpenCV | CLAHE contrast, adaptive thresholding, glare reduction |
| **Flat OCR** | PaddleOCR | Extract text from flat labels with per-line confidence |
| **Cylindrical Vision** | Gemini 2.0 Flash (Vision) | Read curved bottle text natively, returns structured JSON |
| **Structured Extraction** | Gemini 2.0 Flash | Parse PaddleOCR text → JSON fields (flat path) |
| **Text Simplification** | Gemini 2.0 Flash | Medical jargon → 5th-grade plain English |
| **Drug Normalization** | NIH RxNorm API | Brand name → Generic → RxCUI (standardized drug ID) |
| **Vector Database** | ChromaDB | Local vector store; FDA chunks + DDInter data |
| **Embeddings** | BGE-M3 (BAAI/bge-m3) | Text → 1024-dim vectors; runs locally |
| **Agent Framework** | LangChain | Orchestrates 3-tool agentic reasoning |
| **Drug Data** | openFDA Drug Label API | Official FDA labeling: dosage, warnings, interactions |
| **Drug Interactions** | DDInter Database (CSV) | Open-access DDI data with severity + management info |
| **Adverse Events** | openFDA Adverse Events API | FDA-reported adverse event summaries |
| **Web Interface** | Streamlit | Dashboard: upload, results, confidence UI, chat |

---

## Data Sources & APIs

### openFDA (Primary Knowledge Source)
- Drug Labels: `https://api.fda.gov/drug/label.json` — dosage, warnings, interactions, ingredients
- Adverse Events: `https://api.fda.gov/drug/event.json` — reported adverse drug events
- Rate limit: 240 req/min, 120K req/day (free API key)

### NIH RxNorm (Drug Name Normalization)
- `https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug}` — brand → generic → RxCUI

> ⚠️ The NIH RxNav **Drug Interaction API was discontinued in January 2024.** MedLabel uses openFDA label drug_interactions fields + DDInter for interaction data instead.

### DDInter (Drug-Drug Interactions)
- **Primary Source (Kaggle Mirror):** [https://www.kaggle.com/datasets/thedevastator/ddinter-dataset-drug-drug-interactions](https://www.kaggle.com/datasets/thedevastator/ddinter-dataset-drug-drug-interactions) — *Use this if the official site has SSL/certificate issues.*
- **Secondary Source (Zenodo):** [https://zenodo.org/records/5549420](https://zenodo.org/records/5549420)
- **Official Site:** [https://ddinter2.scbdd.com](https://ddinter2.scbdd.com) — *Main reference site for data schema.*
- Place downloaded `*.csv` files in `data/ddinter/` (see `data/ddinter/README.md`).

**Test `interaction_check` locally:**

```bash
export PYTHONPATH=src
python3 scripts/test_interaction_check.py --scanned acetaminophen --query ibuprofen
```

Without CSVs, a small built-in demo pair set is used for development.

---

## What Gets Stored in ChromaDB

Every text chunk in the vector database has metadata for precise filtering:

```python
{
  "document": "Adults and children 12 years and over: take 2 caplets every 6 hours...",
  "metadata": {
    "drug_name": "Acetaminophen",         # generic name
    "brand_name": "Tylenol",              # brand name
    "rxcui": "161",                       # standardized ID
    "section_type": "dosage_and_administration",
    "source": "fda_label",               # fda_label | ddinter | user_scan
    "packaging_type": "flat"             # flat | cylindrical
  }
}
```

When the user scans Tylenol and asks a question, the agent queries **only Acetaminophen chunks** — never accidentally returning Advil information.

### Pre-loaded Starter Set (30-50 drugs)
Common OTC drugs covering all packaging types: Tylenol, Advil, Claritin, Benadryl, Zyrtec, NyQuil, Pepto-Bismol, Tums, Robitussin, Mucinex, Flonase, Neosporin, Aspirin, Melatonin, and more.

**On-demand ingestion:** When a user scans a drug not in the starter set, the app automatically queries openFDA, chunks it, embeds it, and stores it in ChromaDB. It's cached for all future queries.

---

## Agentic Chatbot — 3 Tools

```
Tool 1: vector_search
  → Searches ChromaDB with metadata filtering locked to scanned drug
  → Triggered by: dosage questions, warnings, ingredient questions

Tool 2: interaction_check
  → Searches DDInter + FDA label drug_interactions section
  → Triggered by: "Can I take this with X?"

Tool 3: adverse_events
  → Queries openFDA /drug/event for most-reported effects
  → Triggered by: "What side effects should I watch for?"
```

### System Prompt (The Safety Core)
```
You are the MedLabel Assistant — a safe, concise medical information helper.

RULES:
1. Use ONLY the provided Context. If the answer isn't there, say
   "I don't have enough information — please consult your pharmacist."
2. Simplify medical jargon to a 5th-grade reading level.
3. If asked about exceeding the labeled dose, state the WARNING.
   Do not provide doses above what the label recommends.
4. Always cite the source: [Source: Dosage & Administration — FDA Label]
5. Format as short bullet points.
6. End every response with: "⚠️ This is for informational purposes only.
   Always verify with the physical label or your pharmacist."
```

---

## Sprint Breakdown (7 Weeks)

### Week 1 — Foundation & Data Collection
**S1 & S2:** Collect 100–200 images of medicine packaging (both flat AND cylindrical). Annotate in Roboflow with 2 classes: `flat_label` / `cylindrical_label`.
**S3:** Set up openFDA + RxNorm API clients. Download DDInter CSVs.
**S4:** Set up repo, environments, install all dependencies on all machines.

**Deliverables:** Labeled 2-class dataset. All APIs returning data. All dependencies working.

---

### Week 2 — Vision Pipeline (Both Paths)
**S1 & S2:** Train YOLOv11 on the 2-class dataset (`device='mps'` on M4 Mac). Build routing logic.
**S3:** Build PaddleOCR wrapper with confidence thresholding and post-processing.
**S4:** Build OpenCV preprocessing (contrast, threshold, glare). Build Gemini Vision wrapper.

**Deliverables:** YOLO routing flat vs. cylindrical. PaddleOCR reading flat labels. Gemini Vision reading bottle photos. End-to-end: photo → text (both paths).

---

### Week 3 — Knowledge Base

**S1:** Build FDA ingestion pipeline for starter 30–50 drugs.
**S2:** Embed chunks with BGE-M3 and store in ChromaDB. Build on-demand ingestion.
**S3:** Parse DDInter CSVs into ChromaDB. Build interaction lookup functions.
**S4:** Build Gemini structured extraction prompt (flat path). Build simplification prompt.

**Deliverables:** ChromaDB populated. Extraction and simplification working on 10+ label samples.

---

### Week 4 — Agentic Chatbot

**S1 & S2:** Build LangChain agent with 3 tools. Test tool routing on varied questions.
**S3:** Build adverse_events tool. Wire RxNorm normalization into main pipeline.
**S4:** Write and iterate system prompts. Test safety guardrails: overdose requests, unknown drugs.

**Deliverables:** Agent routing correctly. Answers grounded in FDA data with citations. Safety guardrails holding.

---

### Week 5 — Full Integration

**S1 & S2:** Full pipeline test: flat box → Path A → agent → response. Bottle → Path B → agent → response.
**S3 & S4:** Drug interaction scenarios. Test 10+ known interaction pairs.
**All:** Edge cases: blurry images, no label detected, unsupported drugs, partial labels.

**Deliverables:** Full pipeline working end-to-end for both paths. Interactions working. Edge cases handled.

---

### Week 6 — Streamlit Interface

**S1 & S2:** Build dashboard. Use `@st.cache_resource` on all models (critical for demo performance).
**S3:** Confidence display, path badge ("📦 OCR Pipeline" / "🤖 Vision AI"), color-coded risk indicators.
**S4:** Loading animations with progress text, session state, safety disclaimer banner.

**Deliverables:** Polished web app with all components integrated.

---

### Week 7 — Testing, Demo Prep & Presentation

**S1 & S3:** Test 20+ real medicine packages. Pre-cache demo drugs. Prepare backup recorded video.
**S2 & S4:** Build presentation slides with architecture diagrams, technical challenges, results.
**All:** Rehearse live demo script.

**Deliverables:** Live-demo-ready app. Slides. Clean repository.

---

## Live Demo Script (~5 minutes)

1. **📦 Flat packaging** — upload a Tylenol box → show "📦 OCR Pipeline" badge → extracted fields, simplified summary → chat: *"What's the dose for an 8-year-old?"* → `vector_search` tool
2. **💊 Cylindrical bottle** — upload an Advil bottle → show "🤖 Vision AI" badge → chat: *"Can I take this with my Tylenol?"* → `interaction_check` tool
3. **⚠️ Adverse events** — still using Advil → chat: *"What side effects have people reported?"* → `adverse_events` tool
4. **🔴 Safety guardrail** — any drug → chat: *"Can I take double the dose?"* → agent refuses, shows FDA label warning

---

## Learning Goals

| Category | Skills |
|---|---|
| Computer Vision | Object detection, YOLO training, 2-class routing, image preprocessing |
| OCR | PaddleOCR on real-world flat labels, confidence scoring, post-processing |
| Multi-modal AI | Gemini Vision API, multi-modal prompting, structured output from images |
| NLP / LLMs | Prompt engineering, structured extraction, text simplification, safety guardrails |
| RAG & Vector DBs | Embeddings, ChromaDB, chunking strategies, metadata-filtered retrieval |
| Agentic AI | LangChain agents, multi-tool orchestration, reasoning chains |
| API Integration | openFDA, RxNorm, DDInter — real government data sources |
| Data Engineering | Ingestion pipelines, chunking, on-demand caching |
| Responsible AI | Medical safety guardrails, hallucination prevention, confidence scoring |
| Full-Stack | End-to-end system from CV to deployed web app |

---

## Prerequisite Knowledge

- DSC 40A or equivalent: Python, ML fundamentals, some image processing exposure
- Helpful but not required: OCR, NLP basics, deep learning frameworks, API experience

---

## Safety & Ethical Guardrails

> ⚠️ MedLabel handles medication information. Incorrect outputs have real safety implications.

1. **No hallucination policy** — chatbot uses ONLY retrieved FDA data. If the answer isn't in context, it refers to a pharmacist.
2. **Confidence scoring** — low-confidence OCR reads are visually flagged.
3. **Dose ceiling enforcement** — requests to exceed labeled doses are refused with the FDA warning.
4. **Persistent safety banner** — displayed on every screen: *"This AI can make mistakes. Always verify with the physical label or your pharmacist. This is not a substitute for professional medical advice."*
5. **Scope limitation** — MedLabel is an informational tool, not diagnostic or prescribing.

---

## Datasets & References

| Resource | Link |
|---|---|
| openFDA Drug Labels | https://open.fda.gov/apis/drug/label/ |
| openFDA Adverse Events | https://open.fda.gov/apis/drug/event/ |
| openFDA Bulk Downloads | https://open.fda.gov/apis/downloads/ |
| NIH RxNorm API | https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html |
| DDInter Database (v2) | [Kaggle](https://www.kaggle.com/datasets/thedevastator/ddinter-dataset-drug-drug-interactions) / [Zenodo](https://zenodo.org/records/5549420) |
| Ultralytics YOLOv11 | https://docs.ultralytics.com/ |
| PaddleOCR | https://github.com/PaddlePaddle/PaddleOCR |
| ChromaDB | https://docs.trychroma.com/ |
| BGE-M3 | https://huggingface.co/BAAI/bge-m3 |
| LangChain | https://python.langchain.com/ |
| Roboflow (Annotation) | https://roboflow.com/ |
<div align="center">

```
███╗   ███╗███████╗██████╗ ██╗      █████╗ ██████╗ ███████╗██╗
████╗ ████║██╔════╝██╔══██╗██║     ██╔══██╗██╔══██╗██╔════╝██║
██╔████╔██║█████╗  ██║  ██║██║     ███████║██████╔╝█████╗  ██║
██║╚██╔╝██║██╔══╝  ██║  ██║██║     ██╔══██║██╔══██╗██╔══╝  ██║
██║ ╚═╝ ██║███████╗██████╔╝███████╗██║  ██║██████╔╝███████╗███████╗
╚═╝     ╚═╝╚══════╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚══════╝
```

**Point your phone at any medicine label. Get plain English answers instantly.**

<br/>

[![Live Demo](https://img.shields.io/badge/🚀%20LIVE%20DEMO-medlabel.streamlit.app-C8FF57?style=for-the-badge&labelColor=0d1117&color=C8FF57)](https://medlabel.streamlit.app/)

<br/>

![FDA](https://img.shields.io/badge/FDA%20Grounded-5%2C976%20Chunks-0466C8?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0xIDE1aC0ydi02aDJ2NnptMC04aC0yVjdoMnYyeiIvPjwvc3ZnPg==)
![BGE-M3](https://img.shields.io/badge/BGE--M3-Neural%20Search-7B2FBE?style=flat-square)
![YOLO](https://img.shields.io/badge/YOLOv11-Geometry%20Router-FF6B35?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-E85D4A?style=flat-square)
![Cross‑Encoder](https://img.shields.io/badge/Cross--Encoder-Reranker-2EC4B6?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

</div>

<br/>

---

<div align="center">

## ⚡ What It Does

</div>

Upload any medicine photo. MedLabel handles the rest.

```
📷 PHOTO IN  →  🔍 YOLO ROUTES  →  📝 OCR READS  →  🧠 BGE-M3 RETRIEVES  →  💬 ANSWER OUT
```

Every answer is **retrieved from 5,976 official FDA label chunks** — ranked by a cross-encoder, cited, never invented.

<br/>

---

<div align="center">

## 🗺️ Architecture

</div>

```
                    ┌──────────────────────────────┐
                    │     Any Medicine Photo        │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       YOLO v11 Router        │
                    │  flat_label / cylindrical    │  ← Roboflow-trained
                    └──────┬───────────────┬───────┘
                           │               │
              ┌────────────▼──┐       ┌────▼──────────────┐
              │    Path A     │       │      Path B        │
              │  OpenCV +     │       │   Vision API       │
              │  pytesseract  │       │  (curved bottles)  │
              └────────┬──────┘       └────────┬───────────┘
                       └───────────┬───────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      BGE-M3 Embedding        │  ← 1024-dim vectors
                    │   +  Cross-Encoder Rerank    │  ← precision retrieval
                    └──────────────┬───────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               ▼                   ▼                   ▼
    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
    │  vector_search   │ │interaction_check │ │  adverse_events  │
    │  Dosage, warnings│ │ DDInter + FDA    │ │  openFDA events  │
    │  ingredients     │ │ cross-reference  │ │  data            │
    └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
             └───────────────────┬┘──────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────────┐
                    │         Streamlit UI         │
                    └──────────────────────────────┘
```

<br/>

---

<div align="center">

## 🛠️ Tech Stack

</div>

<table>
<tr>
<td width="50%" valign="top">

**Vision & OCR**
| Component | Technology |
|---|---|
| Geometry Router | YOLOv11 on Roboflow |
| Flat Labels | pytesseract + OpenCV |
| Curved Bottles | Vision API |
| Image I/O | Pillow + pillow-heif |

</td>
<td width="50%" valign="top">

**Intelligence**
| Component | Technology |
|---|---|
| Embeddings | BGE-M3 (1024-dim) |
| Reranker | Cross-Encoder (ST) |
| Vector Store | ChromaDB |
| Drug IDs | NIH RxNorm API |

</td>
</tr>
<tr>
<td width="50%" valign="top">

**Data Sources**
| Source | Content |
|---|---|
| openFDA Labels | Dosage, warnings, interactions |
| DDInter | Drug-drug interaction pairs |
| openFDA Events | Adverse event reports |
| On-demand | Auto-ingest unknown drugs |

</td>
<td width="50%" valign="top">

**Application**
| Component | Technology |
|---|---|
| Web UI | Streamlit |
| Deployment | Streamlit Cloud |
| Secrets | `.streamlit/secrets.toml` |
| Language | Python 3.10+ |

</td>
</tr>
</table>

<br/>

---

<div align="center">

## 💬 The Chatbot — 3 Specialized Tools

</div>

<table>
<tr>
<td align="center" width="33%">

**🔍 vector_search**

Semantic search over FDA label chunks. Handles dosage, warnings, ingredients, and general questions. BGE-M3 retrieves, cross-encoder reranks.

</td>
<td align="center" width="33%">

**⚠️ interaction_check**

Cross-references DDInter + FDA `drug_interactions` sections. Handles drug-drug *and* drug-substance pairs (e.g. alcohol, grapefruit).

</td>
<td align="center" width="33%">

**🩺 adverse_events**

Pulls the most-reported reactions from openFDA's adverse event database — real-world post-market safety data.

</td>
</tr>
</table>

> **Substance fallback** — If you ask *"Can I drink alcohol with this?"*, a text-scan searches every label chunk for the substance, so nothing slips through even when it's not a named drug in the database.

<br/>

---

<div align="center">

## 📦 The Data

</div>

```
ChromaDB
├── 5,976 FDA label chunks  (57+ drugs, BGE-M3 embedded)
├── DDInter interaction pairs  (severity + management)
└── On-demand ingestion  (new drugs auto-fetched from openFDA)

Each chunk:
{
  "drug_name":    "ibuprofen",
  "brand_name":   "Advil",
  "rxcui":        "5640",
  "section_type": "warnings_and_cautions",
  "source":       "fda_label"
}
```

Metadata filtering ensures a question about Advil **only** retrieves Advil chunks — never Tylenol data by accident.

<br/>

---

<div align="center">

## 🚀 Quick Start

</div>

**Clone & run locally:**

```bash
git clone https://github.com/prs-016/MedLabel.git
cd MedLabel

# Full local stack (EasyOCR + all models)
pip install -r requirements-full.txt

# Configure secrets
cp .env.example .env
# → Add XAI_API_KEY, GEMINI_API_KEY, ROBOFLOW_API_KEY

streamlit run app/streamlit_app.py
```

**Lightweight install** (Cloud-compatible, no PyTorch):

```bash
pip install -r requirements.txt
# tesseract binary handled via packages.txt on Streamlit Cloud
```

<br/>

**Or just use the live demo →** [medlabel.streamlit.app](https://medlabel.streamlit.app/)

<br/>

---

<div align="center">

## 🔒 Safety

</div>

MedLabel is an **informational tool** — not diagnostic, not prescriptive.

- ✅ Answers cite only retrieved FDA data — never hallucinated
- ✅ Requests to exceed labeled doses are refused with the FDA warning
- ✅ Low-confidence OCR reads are flagged visually
- ✅ Zero user data stored — no scans, no queries, no logs

> ⚠️ Always verify with the physical label or your pharmacist. This is not a substitute for professional medical advice.

<br/>

---

<div align="center">

## 📚 References

</div>

| Resource | Link |
|---|---|
| openFDA Drug Labels | https://open.fda.gov/apis/drug/label/ |
| openFDA Adverse Events | https://open.fda.gov/apis/drug/event/ |
| NIH RxNorm API | https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html |
| DDInter (Kaggle) | https://www.kaggle.com/datasets/thedevastator/ddinter-dataset-drug-drug-interactions |
| DDInter (Zenodo) | https://zenodo.org/records/5549420 |
| BGE-M3 | https://huggingface.co/BAAI/bge-m3 |
| ChromaDB | https://docs.trychroma.com/ |
| Roboflow | https://roboflow.com/ |

<br/>

---

<div align="center">

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medlabel.streamlit.app/)

<br/>

**Built by the MedLabel team · UC San Diego**

</div>

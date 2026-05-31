# Deploy MedLabel

Two options: **Streamlit Cloud** (persistent public URL) or **local tunnel** (fastest for a demo today).

---

## Option A — Streamlit Community Cloud (recommended)

### 1. Push deploy-ready code

Merge your branch to `main` on GitHub (or deploy from `thaneesh-vector_search`).

Files added for cloud:

- `requirements-cloud.txt` — lighter deps (no Paddle/YOLO)
- `.streamlit/config.toml`
- `src/knowledge/bootstrap.py` — loads DB on startup

### 2. Pack the label database

`medlabel_db/` is gitignored (~175 MB). Cloud needs it via tarball or ingest.

```bash
bash scripts/pack_medlabel_db.sh
```

Upload **`medlabel_db.tar.gz`** to a [GitHub release](https://github.com/prs-016/MedLabel/releases) on the repo.

### 3. Create the app

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → repo `prs-016/MedLabel`, branch `main` (or your branch).
3. **Main file path:** `app/streamlit_app.py`
4. **Advanced settings → Python version:** 3.10 or 3.11
5. **Advanced settings → Dependencies file:** `requirements-cloud.txt`

### 4. Secrets (Settings → Secrets)

Paste (use your real xAI key):

```toml
XAI_API_KEY = "xai-..."
XAI_TEXT_MODEL = "grok-3"
XAI_VISION_MODEL = "grok-4"
CHROMA_DB_PATH = "./medlabel_db"
CHROMA_COLLECTION = "medlabel_chunks"
MEDLABEL_DB_TAR_URL = "https://github.com/prs-016/MedLabel/releases/download/v1.0/medlabel_db.tar.gz"
```

Replace the release URL with your actual asset URL after uploading the tarball.

### 5. Deploy

Click **Deploy**. First boot downloads the DB (~1–2 min), then loads models. You get:

`https://<your-app-name>.streamlit.app`

**Cloud limits:** Free tier has ~1 GB RAM. Chat + RAG works; flat PaddleOCR is not installed on cloud (bottle/xAI Vision OCR still works if `XAI_API_KEY` is set).

---

## Option B — Public link to your Mac (demo today)

While Streamlit is running locally:

```bash
# Install once (pick one)
brew install cloudflared
# or: brew install ngrok

# Tunnel (Streamlit already on :8501)
cloudflared tunnel --url http://localhost:8501
# or: ngrok http 8501
```

Copy the `https://….trycloudflare.com` or `ngrok` URL and share it. Your Mac must stay on and Streamlit running.

No API keys in the URL — they stay in your local `.env`.

---

## Verify after deploy

Ask in chat:

- `What are the side effects of warfarin?` → adverse_events
- `Can I take warfarin with aspirin?` → interaction_check
- `What dose of metformin for adults?` → vector_search

Sidebar should show **~5900+ chunks** in `medlabel_chunks`.

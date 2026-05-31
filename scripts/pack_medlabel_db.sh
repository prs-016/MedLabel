#!/usr/bin/env bash
# Pack medlabel_db for upload to a GitHub release (set MEDLABEL_DB_TAR_URL in Streamlit secrets).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -d medlabel_db ]]; then
  echo "medlabel_db/ not found. Run: PYTHONPATH=src python src/knowledge/ingest_data.py"
  exit 1
fi
tar -czf medlabel_db.tar.gz medlabel_db
ls -lh medlabel_db.tar.gz
echo "Upload medlabel_db.tar.gz to a GitHub release and set MEDLABEL_DB_TAR_URL in Streamlit secrets."

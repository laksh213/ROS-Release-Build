#!/usr/bin/env bash
# ROScribe — full corpus build. RESUMABLE: safe to re-run any time; it skips
# work already done and picks up where an interrupted run stopped.
#
#   ./scripts/build_all.sh          # run / resume the whole pipeline
#   tail -f /tmp/roscribe_build.log # watch progress
#
# Stages: scrape all judgements -> embed judgements (bge-m3) -> embed notes.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

# Multilingual bge-m3 for the full corpus (overrides .env for this run).
export ROSCRIBE_EMBEDDER="${ROSCRIBE_EMBEDDER:-bge}"
NOTES="${PERSONAL_REPO_DIR:-/Users/laksh/Desktop/Final Year April 2026}"

echo "[$(date)]  1/3  scrape all judgements (resumable)"
python -m src.scrape || echo "  scrape exited $? — continuing with what downloaded"

echo "[$(date)]  2/3  index judgements on bge-m3 (resumable, GPU/MPS)"
python -m src.index

echo "[$(date)]  3/3  index notes on bge-m3 (resumable)"
python -m src.index --notes "$NOTES"

# Point the app at the full multilingual index.
sed -i '' 's/^ROSCRIBE_EMBEDDER=.*/ROSCRIBE_EMBEDDER=bge/' .env 2>/dev/null || true
echo "[$(date)]  DONE.  Restart the UI:  streamlit run app/streamlit_app.py"

#!/usr/bin/env bash
# Launch the UI using the prepared document embeddings and index.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for artifact in chunks.jsonl embeddings.npz index.faiss; do
    if [[ ! -s "$PROJECT_ROOT/data/$artifact" ]]; then
        echo "Missing data/$artifact. Get the prepared data files included with the project."
        exit 1
    fi
done

if [[ -f "$PROJECT_ROOT/.venv/Scripts/python.exe" ]]; then
    PROJECT_PYTHON="$PROJECT_ROOT/.venv/Scripts/python.exe"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
    echo "Create the Python environment and install dependencies using the README first."
    exit 1
fi

exec "$PROJECT_PYTHON" -X utf8 -m streamlit run "$PROJECT_ROOT/scripts/app.py" \
    --server.address 127.0.0.1 --theme.base light \
    --theme.backgroundColor '#fbf8f3' --theme.secondaryBackgroundColor '#f3ece2' \
    --theme.textColor '#354b48' --theme.primaryColor '#386f69' "$@"

#!/usr/bin/env bash
#
# run.sh — one-command runner for the BirdCLEF 2026 autonomous agents.
#
# Sets up the environment (deps + Ollama), then runs the FULL research loop for
# BOTH backbones in sequence:
#     1. EfficientNet   (repo root / agent.py)
#     2. YamNet         ("Yamnet runs/agent.py")
#
# Usage:
#     conda activate keras_env     # the environment you trained in
#     ./run.sh
#
# Notes:
#   * This is the full run (DEBUG=False in both agents) — expect SEVERAL HOURS
#     per backbone on CPU. Output is streamed to the console and saved to logs/.
#   * Requires Ollama (https://ollama.com) for the local LLM; the script starts
#     the server and pulls the model automatically.
#   * Installs into whatever Python is currently active — activate your env first.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LLM_MODEL="gemma4:e4b"          # must match LLM_MODEL in both agent.py files
YAMNET_DIR="Yamnet runs"
LOG_DIR="$ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

bold() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

# ── Setup steps abort on failure ────────────────────────────────────────────
set -e

bold "[1/5] Installing Python dependencies (pinned TF 2.19.0 / Keras 3.10.0)"
python -m pip install --upgrade pip || true
python -m pip install -r requirements.txt

bold "[2/5] Verifying TensorFlow / Keras versions (agents refuse to run on a mismatch)"
python - <<'PY'
import tensorflow as tf, keras
print("TensorFlow", tf.__version__, "| Keras", keras.__version__)
assert tf.__version__ == "2.19.0",  "TensorFlow must be 2.19.0 to match Kaggle"
assert keras.__version__ == "3.10.0", "Keras must be 3.10.0 to match Kaggle"
print("Versions OK.")
PY

bold "[3/5] Ensuring Ollama is running and '$LLM_MODEL' is pulled"
if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: ollama is not installed. Get it from https://ollama.com, then re-run." >&2
    exit 1
fi
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "    starting 'ollama serve' (background; log: logs/ollama_$STAMP.log)"
    nohup ollama serve >"$LOG_DIR/ollama_$STAMP.log" 2>&1 &
    for _ in $(seq 1 30); do
        curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
    done
fi
ollama pull "$LLM_MODEL"

bold "[4/5] Linking shared data/ into '$YAMNET_DIR/'"
# The YamNet agent resolves data relative to its own folder, but the dataset
# lives only at the repo root. Link it in (idempotent — no data is copied).
if [ ! -e "$YAMNET_DIR/data" ]; then
    ln -s ../data "$YAMNET_DIR/data"
    echo "    created symlink '$YAMNET_DIR/data' -> ../data"
else
    echo "    '$YAMNET_DIR/data' already present"
fi

# ── Agent runs continue independently (one failing must not skip the other) ──
set +e

bold "[5/5] Running agents — full research run (this takes hours)"

run_one () {                       # $1 = label, $2 = working dir
    local label="$1" dir="$2"
    local log="$LOG_DIR/${label}_$STAMP.log"
    echo
    echo "──────── ${label} agent — started $(date '+%Y-%m-%d %H:%M:%S') ────────"
    echo "         streaming to $log"
    if ( cd "$dir" && python agent.py ) 2>&1 | tee "$log"; then
        echo "✓ ${label} agent finished."
    else
        echo "✗ ${label} agent exited with an error — see $log (continuing)." >&2
    fi
}

run_one "efficientnet" "$ROOT"
run_one "yamnet"       "$ROOT/$YAMNET_DIR"

bold "Done — EfficientNet + YamNet runs complete."
echo "Results:  runs/  and  '$YAMNET_DIR/runs/'        Logs:  logs/"

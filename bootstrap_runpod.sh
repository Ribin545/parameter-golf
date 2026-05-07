#!/usr/bin/env bash
# =============================================================================
# bootstrap_runpod.sh — Idempotent RunPod setup + AB test launcher
# ---------------------------------------------------------------------------
# Safe to run multiple times. If repo exists → git pull; else → git clone.
# Then installs deps and runs the AB test.
#
# One-liner (from /workspace):
#   curl -sSL https://raw.githubusercontent.com/Ribin545/parameter-golf/master/bootstrap_runpod.sh | bash
# =============================================================================
set -euo pipefail

REPO_DIR="/workspace/parameter-golf"
REPO_URL="https://github.com/Ribin545/parameter-golf.git"

echo "=========================================="
echo " RunPod Bootstrap — Parameter Golf AB Test"
echo "=========================================="

# --- Step 1: Clone or pull repo ---
if [ -d "$REPO_DIR/.git" ]; then
    echo "[bootstrap] Repo already exists. Pulling latest..."
    cd "$REPO_DIR"
    git fetch origin
    git reset --hard origin/master
    git pull origin master
else
    if [ -d "$REPO_DIR" ]; then
        echo "[bootstrap] Directory exists but no .git — removing and re-cloning..."
        rm -rf "$REPO_DIR"
    fi
    echo "[bootstrap] Cloning repo..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# --- Step 2: Install dependencies ---
echo "[bootstrap] Installing Python dependencies..."
pip install -r requirements.txt --quiet

# --- Step 3: Run AB test ---
echo "[bootstrap] Launching AB test..."
chmod +x run_ab_test.sh
./run_ab_test.sh

echo ""
echo "=========================================="
echo " Bootstrap complete."
echo " Logs: $REPO_DIR/ab_logs/"
echo "=========================================="
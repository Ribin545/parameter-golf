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
    git checkout -B master origin/master
else
    if [ -d "$REPO_DIR" ]; then
        echo "[bootstrap] Directory exists but no .git — removing and re-cloning..."
        rm -rf "$REPO_DIR"
    fi
    echo "[bootstrap] Cloning repo..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    git fetch origin
    git checkout -B master origin/master
fi

echo "[bootstrap] Repo HEAD: $(git rev-parse --short HEAD)"
ls -1

if [ ! -f trial_5090.sh ]; then
    echo "[bootstrap][fatal] trial_5090.sh missing after checkout"
    exit 1
fi

# --- Step 2: Install PyTorch with pinned version ---
echo "[bootstrap] Installing PyTorch nightly (cu130 for RTX 5090 Blackwell sm_120)..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

# --- Step 3: Install remaining dependencies ---
echo "[bootstrap] Installing Python dependencies..."
pip install -r requirements.txt --quiet

# --- Step 4: Run 5090 throughput training ---
echo "[bootstrap] Launching 5090 throughput training..."
chmod +x trial_5090.sh
./trial_5090.sh

echo ""
echo "=========================================="
echo " Bootstrap complete."
echo " Logs: $REPO_DIR/ab_logs/"
echo "=========================================="
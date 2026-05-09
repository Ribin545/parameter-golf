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

# --- Step 2: Install remaining dependencies first (except torch stack) ---
echo "[bootstrap] Installing Python dependencies..."
python3 -m pip install --upgrade pip --quiet
python3 -m pip install numpy tqdm huggingface-hub kernels setuptools typing-extensions==4.15.0 datasets tiktoken sentencepiece triton --quiet

# --- Step 3: Force-install exact PyTorch build LAST so nothing overwrites it ---
echo "[bootstrap] Removing incompatible preinstalled torch packages..."
python3 -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
echo "[bootstrap] Installing exact PyTorch nightly cu130 (Blackwell sm_120 support for RTX 5090)..."
python3 -m pip install --no-cache-dir --force-reinstall torch==2.13.0.dev20260507 --index-url https://download.pytorch.org/whl/nightly/cu130
python3 - <<'PY'
import torch
print('[bootstrap] torch_check', torch.__version__, 'cuda=', torch.version.cuda)
PY

# --- Step 4: Run 5090 throughput training ---
echo "[bootstrap] Launching 5090 throughput training..."
chmod +x trial_5090.sh
./trial_5090.sh

echo ""
echo "=========================================="
echo " Bootstrap complete."
echo " Logs: $REPO_DIR/ab_logs/"
echo "=========================================="
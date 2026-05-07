#!/usr/bin/env bash
# =============================================================================
# run_ab_test.sh — AB Test Phase 1 on Linux (with Triton/inductor for speed)
# Usage:
#   chmod +x run_ab_test.sh
#   ./run_ab_test.sh
#
# Assumes dataset at $DATA_PATH (or default relative path).
# Uses native torch.compile with inductor+Triton for sub-600ms step times.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default data paths
export DATA_PATH="${DATA_PATH:-${SCRIPT_DIR}/../../../data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${SCRIPT_DIR}/../../../data/tokenizers/fineweb_1024_bpe.model}"

# Check dataset exists
if [ ! -d "$DATA_PATH" ]; then
    echo "[ERROR] Dataset not found at: $DATA_PATH"
    echo "  Set DATA_PATH env var or download:"
    echo "    git clone https://github.com/openai/parameter-golf.git /tmp/pg"
    echo "    python3 /tmp/pg/data/cached_challenge_fineweb.py --variant sp1024"
    echo "    mkdir -p $(dirname "$DATA_PATH")"
    echo "    mv /tmp/pg/data/datasets/fineweb10B_sp1024 $DATA_PATH"
    echo "    mv /tmp/pg/data/tokenizers/fineweb_1024_bpe.model $TOKENIZER_PATH"
    exit 1
fi

# Verify CUDA + Triton
echo "=== Environment ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem // (1024**3)} GiB')
try:
    import triton
    print(f'Triton: {triton.__version__}')
except ImportError:
    print('Triton: NOT INSTALLED (pip install triton)')
"
echo ""

echo "=========================================================================="
echo "AB TEST Phase 1 — Linux (inductor + Triton)"
echo "  Config A: Recurrent single-block x12 steps"
echo "  Config B: 11 distinct layers (shared QKV weights)"
echo "  75 iterations each, deterministic data"
echo "========================================================================"
echo ""

# Run both configs via ab_test.py (platform-aware — uses train_gpt.py directly)
python3 -u ab_test.py

echo ""
echo "=== AB Test Complete ==="
echo "Logs: ab_logs/config_a.txt  ab_logs/config_b.txt"
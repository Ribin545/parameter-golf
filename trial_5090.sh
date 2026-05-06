#!/usr/bin/env bash
# =============================================================================
# trial_5090.sh — ScaleDown config for Linux (RTX 5090 / RunPod trial)
# Usage:
#   chmod +x trial_5090.sh
#   ./trial_5090.sh              # full 10-min baseline
#   ./trial_5090.sh --smoke      # 60-sec smoketest only
#   ./trial_5090.sh --both       # smoketest then full baseline
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse mode
MODE="${1:-full}"
CONFIG="${2:-}"
case "$MODE" in
    --smoke|smoke)
        MODE="smoke"
        ;;
    --both|both)
        MODE="both"
        ;;
    --gate|--speed-gate|gate)
        MODE="gate"
        ;;
    *)
        MODE="full"
        ;;
esac

# =============================================================================
# Default paths — override via env if your RunPod setup differs
# =============================================================================
export DATA_PATH="${DATA_PATH:-${SCRIPT_DIR}/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${SCRIPT_DIR}/data/tokenizers/fineweb_1024_bpe.model}"
: "${VOCAB_SIZE:=1024}"

# Check data exists
if [ ! -d "$DATA_PATH" ]; then
    echo "[ERROR] Dataset not found at: $DATA_PATH"
    echo "  Download it first:"
    echo "    git clone https://github.com/openai/parameter-golf.git /tmp/pg"
    echo "    python3 /tmp/pg/data/cached_challenge_fineweb.py --variant sp1024"
    echo "    mkdir -p data/datasets data/tokenizers"
    echo "    mv /tmp/pg/data/datasets/fineweb10B_sp1024 $DATA_PATH"
    echo "    mv /tmp/pg/data/tokenizers/fineweb_1024_bpe.model $TOKENIZER_PATH"
    exit 1
fi

# =============================================================================
# Architecture (same as ScaleDown.bat)
# =============================================================================
export MODEL_DIM=1024
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=5
export RECURRENCE_STEPS=1
export LORA_RANK=512
export LORA_SCOPE=q

# =============================================================================
# Training batch settings
# =============================================================================
export TRAIN_BATCH_TOKENS=524288
export GRAD_ACCUM_STEPS=16
export TRAIN_SEQ_LEN=1024

# =============================================================================
# Learning rates
# =============================================================================
export MATRIX_LR=0.08
export SCALAR_LR=0.015
export LORA_LR=0.015
export CONTROL_LR=0.015
export EMBED_LR=0.3
export HEAD_LR=0.008
export TIED_EMBED_LR=0.06

# =============================================================================
# Weight decay
# =============================================================================
export SCALAR_WEIGHT_DECAY=0.1
export LORA_WEIGHT_DECAY=0.0
export CONTROL_WEIGHT_DECAY=0.0

# =============================================================================
# Gradients
# =============================================================================
export DYNAMIC_LR_NORM=1
export TARGET_GRAD_NORM=0.5
export GRAD_CLIP_NORM=1.0
export MUON_BACKEND_STEPS=5
export QK_GAIN_INIT=1.5

# =============================================================================
# Loss filter
# =============================================================================
export LOSS_FILTER_ENABLED=1
export LOSS_FILTER_WARMUP=250
export LOSS_FILTER_MAX_RECENT_DROP=0.02
export LOSS_FILTER_Z_THRESHOLD=4.5

# =============================================================================
# Shell centering
# =============================================================================
export SHELL_CENTERING_ENABLED=1
export SHELL_CENTERING_LAM=0.008

# =============================================================================
# Features
# =============================================================================
export BIGRAM_HASH_ENABLED=1
export BIGRAM_HASH_SIZE=2048
export BIGRAM_HASH_SCALE=0.05
export LEVEL_SIGNAL_ENABLED=0
export TTT_ENABLED=0

# =============================================================================
# Data determinism
# =============================================================================
export DATA_DETERMINISTIC=1
export DATA_SEED=3623123517
export VAL_LOSS_EVERY=100
export WARMUP_STEPS=16
export SEQ_LEN_CURRICULUM=0
export RECURRENCE_CURRICULUM=0

# =============================================================================
# Run
# =============================================================================
run_training() {
    local label="$1"
    local wallclock="$2"
    local iters="$3"

    echo ""
    echo "=========================================================================="
    echo "  $label"
    echo "  Wallclock: ${wallclock}s  |  Max iters: $iters"
    echo "  MODEL_DIM=$MODEL_DIM  MLP_MULT=$MLP_MULT  LORA_SCOPE=$LORA_SCOPE"
    echo "  ShellCentering=$SHELL_CENTERING_ENABLED (λ=$SHELL_CENTERING_LAM)"
    echo "  DYNORM=$DYNAMIC_LR_NORM (target=$TARGET_GRAD_NORM)  LossFilter=$LOSS_FILTER_ENABLED"
    echo "  MATRIX_LR=$MATRIX_LR  EMBED_LR=$EMBED_LR  HEAD_LR=$HEAD_LR"
    echo "=========================================================================="

    export MAX_WALLCLOCK_SECONDS=$wallclock
    export ITERATIONS=$iters

    python3 train_gpt.py
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "[FAILED] $label — exit code $exit_code"
        return 1
    fi
    echo "[PASSED] $label"
    return 0
}

# Check Python / CUDA
echo "=== Environment Check ==="
python3 -c "import torch; print(f'PyTorch {torch.__version__}  |  CUDA available: {torch.cuda.is_available()}  |  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || {
    echo "[FATAL] PyTorch not found. pip install -r requirements.txt"
    exit 1
}
echo ""

# =============================================================================
# Speed-gate mode — lean config for 75-iter AB tests
# =============================================================================
run_gate() {
    local tag="${1:-lean}"
    echo ""
    echo "=========================================================================="
    echo "  PHASE 1 GATE: $tag (75 iters, no compile)"
    echo "=========================================================================="

    # Override architecture for speed
    if [ "$tag" = "512" ]; then
        export MODEL_DIM=512
        export NUM_HEADS=4
        export NUM_KV_HEADS=2
        export MLP_MULT=8
        export LORA_RANK=128
    fi

    # Speed knobs
    export DISABLE_COMPILE=1
    export ITERATIONS=75
    export MAX_WALLCLOCK_SECONDS=9999
    export DATA_DETERMINISTIC=1
    export DATA_SEED=3623123517
    export LOSS_FILTER_ENABLED=0   # waste of time at 75 iters
    export VAL_LOSS_EVERY=25
    export WARMUP_STEPS=4

    python3 train_gpt.py
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAILED] GATE: $tag — exit code $rc"
        return 1
    fi
    echo "[PASSED] GATE: $tag"
}

if [ "$MODE" = "gate" ]; then
    run_gate "1024"
    run_gate "512"
    exit 0
fi

if [ "$MODE" = "smoke" ] || [ "$MODE" = "both" ]; then
    run_training "SMOKETEST (60s)" 60 50 || {
        echo "[FATAL] Smoketest failed — fix issues before running baseline"
        exit 1
    }
fi

if [ "$MODE" = "full" ] || [ "$MODE" = "both" ]; then
    run_training "BASELINE (600s)" 600 415 || {
        echo "[FATAL] Baseline failed"
        exit 1
    }
fi

echo ""
echo "=== All Done ==="
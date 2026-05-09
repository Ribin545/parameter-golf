#!/usr/bin/env bash
# =============================================================================
# trial_5090.sh — Optimized Multilayer 5L×2S for RTX 5090 / RunPod
# Usage:
#   chmod +x trial_5090.sh
#   ./trial_5090.sh              # full 10-min run
#   ./trial_5090.sh --smoke      # 60-sec smoketest
#   ./trial_5090.sh --smoke 131k # smoketest with bigger micro-batch
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse mode + optional micro-batch override
MODE="${1:-full}"
MB_OVERRIDE="${2:-}"
case "$MODE" in
    --smoke|smoke)  MODE="smoke" ;;
    *)              MODE="full" ;;
esac

# =============================================================================
# Data paths
# =============================================================================
export DATA_PATH="${DATA_PATH:-${SCRIPT_DIR}/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${SCRIPT_DIR}/data/tokenizers/fineweb_1024_bpe.model}"

if [ ! -d "$DATA_PATH" ]; then
    echo "[setup] Dataset not found. Downloading FineWeb-10B (sp1024) ..."
    echo "[setup] This is ~20 GB and may take 5-15 minutes."
    
    if [ ! -d /tmp/pg ]; then
        git clone --depth 1 https://github.com/openai/parameter-golf.git /tmp/pg
    fi
    python3 /tmp/pg/data/cached_challenge_fineweb.py --variant sp1024
    mkdir -p "$SCRIPT_DIR/data/datasets" "$SCRIPT_DIR/data/tokenizers"
    cp -r /tmp/pg/data/datasets/fineweb10B_sp1024 "$SCRIPT_DIR/data/datasets/"
    cp /tmp/pg/data/tokenizers/fineweb_1024_bpe.model "$SCRIPT_DIR/data/tokenizers/"
    
    echo "[setup] Dataset download complete."
fi

# =============================================================================
# Architecture: Multilayer U-Net 5L×2S (384-dim with LoRA)
# =============================================================================
export MODEL_TYPE=multilayer
export NUM_LAYERS=5
export MODEL_DIM=384
export NUM_HEADS=4
export NUM_KV_HEADS=2
export MLP_MULT=2
export RECURRENCE_STEPS=2
export MULTILAYER_LORA_RANK=256

# =============================================================================
# Batch: 524k tokens/step — micro-batch reduced for VRAM safety (618 MB headroom)
# =============================================================================
export MICRO_BATCH_TOKENS="${MB_OVERRIDE:-131072}"
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024

# =============================================================================
# Optimizer: Muon + AdamW — aggressively tuned for fast convergence
# =============================================================================
export OPTIM_MODE=muon_adam
export MATRIX_OPTIM=muon
export MATRIX_LR=0.12
export SCALAR_LR=0.03
export LORA_LR=0.03
export CONTROL_LR=0.03
export EMBED_LR=0.3
export HEAD_LR=0.008
export TIED_EMBED_LR=0.03
export MUON_BACKEND_STEPS=3
export MUON_MOMENTUM=0.95

# =============================================================================
# Weight decay
# =============================================================================
export SCALAR_WEIGHT_DECAY=0.2
export LORA_WEIGHT_DECAY=0.0
export CONTROL_WEIGHT_DECAY=0.0

# =============================================================================
# Gradients
# =============================================================================
export DYNAMIC_LR_NORM=1
export TARGET_GRAD_NORM=0.5
export GRAD_CLIP_NORM=1.0
export QK_GAIN_INIT=4.0

# =============================================================================
# torch.compile — DISABLED (CUDA graph cache sharing is unsafe under grad acc)
# =============================================================================
export DISABLE_COMPILE=1

# =============================================================================
# Features (quality knobs — cheap enough to keep)
# =============================================================================
export BIGRAM_HASH_ENABLED=0
export BIGRAM_HASH_SIZE=2048
export BIGRAM_HASH_SCALE=0.05
export SHELL_CENTERING_ENABLED=1
export SHELL_CENTERING_LAM=0.008
export TIE_EMBEDDINGS=1
export TIED_EMBED_INIT_STD=0.005

# =============================================================================
# Regularization
# =============================================================================
export DROPOUT_P=0.4
export LABEL_SMOOTHING=0.15

# =============================================================================
# Scheduler + stopping
# =============================================================================
export WARMUP_STEPS=10
export VAL_LOSS_EVERY=200
export SAVE_BEST_CHECKPOINT=1
export SAVE_BEST_INT8=1
export EXPORT_BEST_CHECKPOINT=1
export SEQ_LEN_CURRICULUM=0
export RECURRENCE_CURRICULUM=0

# =============================================================================
# Data determinism
# =============================================================================
export DATA_DETERMINISTIC=1
export DATA_SEED=3623123517
export VOCAB_SIZE=1024

# =============================================================================
# Logger
# =============================================================================
run_training() {
    local label="$1"
    local wallclock="$2"
    local iters="$3"

    echo ""
    echo "=========================================================================="
    echo "  $label"
    echo "  MODEL_TYPE=$MODEL_TYPE  LAYERS=$NUM_LAYERS  DIM=$MODEL_DIM  STEPS=$RECURRENCE_STEPS"
    echo "  HEADS=$NUM_HEADS  KV_HEADS=$NUM_KV_HEADS  MLP_MULT=$MLP_MULT"
    echo "  BATCH: ${TRAIN_BATCH_TOKENS} tokens/step  (micro=${MICRO_BATCH_TOKENS})"
    echo "  OPTIM: $OPTIM_MODE  MuonNS=$MUON_BACKEND_STEPS"
    echo "  FEATURES: Bigram=$BIGRAM_HASH_ENABLED  ShellCentering=$SHELL_CENTERING_ENABLED"
    echo "  Wallclock: ${wallclock}s  |  Max iters: $iters"
    echo "  torch.compile: $(if [ "${DISABLE_COMPILE:-0}" = "1" ]; then echo OFF; else echo "ON (default)"; fi)"
    echo "=========================================================================="

    export MAX_WALLCLOCK_SECONDS=$wallclock
    export ITERATIONS=$iters

    python3 -u train_gpt.py
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "[FAILED] $label — exit code $exit_code"
        return 1
    fi
    echo "[PASSED] $label"
    return 0
}

# =============================================================================
# Environment check
# =============================================================================
echo "=== Environment Check ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    print(f'GPU: {gpu_name}  ({vram_gb} GiB)')
try:
    import triton
    print(f'Triton: {triton.__version__}')
except ImportError:
    print('Triton: NOT INSTALLED')
" || {
    echo "[FATAL] PyTorch not found. pip install -r requirements.txt"
    exit 1
}
echo ""

# =============================================================================
# Run
# =============================================================================
if [ "$MODE" = "smoke" ]; then
    run_training "SMOKETEST (60s)" 60 50 || {
        echo "[FATAL] Smoketest failed"
        exit 1
    }
else
    run_training "5090 THROUGHPUT (600s)" 600 999999 || {
        echo "[FATAL] Full run failed"
        exit 1
    }
fi

echo ""
echo "=== All Done ==="
echo "Model exported to: final_model.int8.ptz"
ls -lh final_model.int8.ptz 2>/dev/null || echo "(checkpoint not found — check logs)"
#!/usr/bin/env bash
# =============================================================================
# trial_h100.sh — H100 Step 1 single-change progression run
# Goal: keep the historical winner recipe intact and change ONLY:
#   MULTILAYER_ACTIVATION_CHECKPOINT_MODE=encoder_only -> minimal
# This preserves the existing bootstrap flow while testing one H100 change.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export DATA_PATH="${DATA_PATH:-${SCRIPT_DIR}/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${SCRIPT_DIR}/data/tokenizers/fineweb_1024_bpe.model}"

# Core architecture (same proven winner family)
export MODEL_TYPE=multilayer
export NUM_LAYERS=5
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=2
export RECURRENCE_STEPS=2
export MULTILAYER_LORA_RANK=8

# H100 Step 3: push VRAM even harder — 524k batch, aggressive compile, no safety limits
export SAFETY_CLAMP_DISABLE=1
export MICRO_BATCH_TOKENS=524288
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024

# Optimizer / training recipe from winner family
export OPTIM_MODE=muon_adam
export MATRIX_OPTIM=muon
export MATRIX_LR=0.15
export SCALAR_LR=0.04
export LORA_LR=0.04
export CONTROL_LR=0.04
export EMBED_LR=0.3
export HEAD_LR=0.008
export TIED_EMBED_LR=0.03
export BETA1=0.9
export MUON_BACKEND_STEPS=7
export MUON_MOMENTUM=0.95
export BETA2=0.95
export SCALAR_WEIGHT_DECAY=0.20
export LORA_WEIGHT_DECAY=0.0
export CONTROL_WEIGHT_DECAY=0.0
export DYNAMIC_LR_NORM=1
export GRAD_CLIP_NORM=1.0
export TARGET_GRAD_NORM=0.8

# Attention / logits / recurrence winner knobs
export QK_GAIN_INIT=3.0
export LOGIT_SOFTCAP=30.0
export Z_LOSS_LAMBDA=0.0
export ATTN_OUTPUT_MODE=einsum_fused
export QK_POST_MODE=baseline
export RECURRENT_ATTN_EVERY=2
export BIGRAM_HASH_ENABLED=1
export BIGRAM_HASH_SIZE=4096
export BIGRAM_HASH_SCALE=0.05
export SHELL_CENTERING_ENABLED=1
export SHELL_CENTERING_LAM=0.005
export TIE_EMBEDDINGS=0
export TIED_EMBED_INIT_STD=0.005
export DROPOUT_P=0.30
export LABEL_SMOOTHING=0.08
export SCHEDULE_FREE=1
export WARMUP_STEPS=40

# Compile / backend — default is the only safe mode with our recurrence structure
export DISABLE_COMPILE=0
export TORCH_COMPILE_MODE=default
export SDPA_BACKEND=flash

# Static mini-depth winner
export MINI_DEPTH_STATIC=1
export MINI_DEPTH_REFINE_BLOCKS=3

# Step 2: push VRAM — disable ALL checkpointing/recompute for minimum overhead
export MULTILAYER_ACTIVATION_CHECKPOINT=0
export MULTILAYER_ACTIVATION_CHECKPOINT_MODE=off
export MLP_MEMORY_MODE=off
export ATTN_MEMORY_MODE=off
export MLP_RECOMPUTE=0

# Data / eval / export
export DATA_DETERMINISTIC=1
export DATA_SEED=3623123517
export VOCAB_SIZE=1024
export TRAIN_LOG_EVERY=50
export VAL_LOSS_EVERY=100
export QUANT_EVAL=1
export SAVE_BEST_CHECKPOINT=1
export SAVE_BEST_INT8=1
export EXPORT_BEST_CHECKPOINT=1
export ITERATIONS=999999
export MAX_WALLCLOCK_SECONDS=600

echo "=========================================================================="
echo "  H100 STEP 3 (push VRAM even harder)"
echo "  NO checkpointing, NO recompute, batch 524288, compile=default"
echo "  MODEL_TYPE=$MODEL_TYPE  LAYERS=$NUM_LAYERS  DIM=$MODEL_DIM  STEPS=$RECURRENCE_STEPS"
echo "  HEADS=$NUM_HEADS  KV_HEADS=$NUM_KV_HEADS  MLP_MULT=$MLP_MULT"
echo "  BATCH: ${TRAIN_BATCH_TOKENS} tokens/step  (micro=${MICRO_BATCH_TOKENS})"
echo "  CHECKPOINTING: enabled=${MULTILAYER_ACTIVATION_CHECKPOINT} mode=${MULTILAYER_ACTIVATION_CHECKPOINT_MODE}"
echo "  MLP_RECOMPUTE=$MLP_RECOMPUTE"
echo "  torch.compile: ON ($TORCH_COMPILE_MODE)"
echo "=========================================================================="

python3 -u train_gpt.py
#!/usr/bin/env bash
# sweep_checkpointing.sh — Activation Checkpointing Mode Sweep (10-min each)
# Tests which checkpoint mode saves the most VRAM, allowing bigger batches.
# Runs sequentially: encoder → full → decoder → alternate → encoder_grouped → off
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# Shared base config (same as trial_5090.sh locked config)
# =============================================================================
export DATA_PATH=/mnt/e/Projects/Proj/golf/data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=/mnt/e/Projects/Proj/golf/data/tokenizers/fineweb_1024_bpe.model
export MODEL_TYPE=multilayer NUM_LAYERS=5 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
export RECURRENCE_STEPS=2 MULTILAYER_LORA_RANK=8
export SAFETY_CLAMP_DISABLE=1
export MICRO_BATCH_TOKENS=204800 TRAIN_BATCH_TOKENS=204800 TRAIN_SEQ_LEN=1024
export OPTIM_MODE=muon_adam MATRIX_OPTIM=muon MATRIX_LR=0.12 SCALAR_LR=0.03
export LORA_LR=0.03 CONTROL_LR=0.03 EMBED_LR=0.3 HEAD_LR=0.008 TIED_EMBED_LR=0.03
export MUON_BACKEND_STEPS=5 MUON_MOMENTUM=0.95 BETA2=0.92
export SCALAR_WEIGHT_DECAY=0.20 LORA_WEIGHT_DECAY=0.0 CONTROL_WEIGHT_DECAY=0.0
export DYNAMIC_LR_NORM=1 TARGET_GRAD_NORM=0.5 GRAD_CLIP_NORM=1.0 DISABLE_COMPILE=1
export SCHEDULE_FREE=1 WARMUP_STEPS=20 TRAIN_LOG_EVERY=10 VAL_LOSS_EVERY=200
export SAVE_BEST_CHECKPOINT=0 SAVE_BEST_INT8=0 EXPORT_BEST_CHECKPOINT=0 QUANT_EVAL=0
export SEQ_LEN_CURRICULUM=0 RECURRENCE_CURRICULUM=0
export DATA_DETERMINISTIC=1 DATA_SEED=3623123517 VOCAB_SIZE=1024
export MAX_WALLCLOCK_SECONDS=600 ITERATIONS=999999
export RECURRENT_ATTN_EVERY=2
export BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_SIZE=4096 BIGRAM_HASH_SCALE=0.05
export SHELL_CENTERING_ENABLED=1 SHELL_CENTERING_LAM=0.008
export TIE_EMBEDDINGS=0 TIED_EMBED_INIT_STD=0.005
export DROPOUT_P=0.4 LABEL_SMOOTHING=0.15
export LOGIT_SOFTCAP=30.0 QK_GAIN_INIT=3.0
export Z_LOSS_LAMBDA=0.0
export LEVEL_SIGNAL_ENABLED=0

PYTHON_BIN="${PYTHON_BIN:-/home/tltp2128/.venvs/pg3090/bin/python}"
LOG_DIR="/tmp/checkpoint_sweep_200k"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.txt"

echo "=== Activation Checkpointing Sweep ===" | tee "$SUMMARY"
echo "Date: $(date)" | tee -a "$SUMMARY"
echo "Batch: 200k tokens/step (micro=TRAIN_BATCH_TOKENS)" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "Mode | val_bpb(S200) | val_bpb(S400) | val_bpb(S600) | val_bpb(final) | step_time_ms | peak_alloc_gib | final_step | notes" | tee -a "$SUMMARY"
echo "-----|--------------|--------------|--------------|--------------|-------------|---------------|------------|------" | tee -a "$SUMMARY"

run_checkpoint_test() {
    local label="$1"
    local ckpt_enabled="$2"
    local ckpt_mode="$3"
    local logfile="$LOG_DIR/${label}.log"

    export MULTILAYER_ACTIVATION_CHECKPOINT=$ckpt_enabled
    export MULTILAYER_ACTIVATION_CHECKPOINT_MODE=$ckpt_mode

    echo "" | tee -a "$SUMMARY"
    echo ">>> [$label] checkpoint=$ckpt_enabled mode=$ckpt_mode" | tee -a "$SUMMARY"

    $PYTHON_BIN -u train_gpt.py > "$logfile" 2>&1
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo ">>> [$label] CRASHED (exit $exit_code)" | tee -a "$SUMMARY"
        echo "$label | - | - | - | CRASHED | - | - | - | exit=$exit_code" | tee -a "$SUMMARY"
        return 1
    fi

    # Parse metrics
    local s200=$(grep -oP 'step:200.*?val_bpb:\K\S+' "$logfile" | tail -1 || echo "-")
    local s400=$(grep -oP 'step:400.*?val_bpb:\K\S+' "$logfile" | tail -1 || echo "-")
    local s600=$(grep -oP 'step:600.*?val_bpb:\K\S+' "$logfile" | tail -1 || echo "-")
    local final_bpb=$(grep -oP 'FINAL STRIDE 64.*?val_bpb:\K\S+' "$logfile" | tail -1 || echo "-")
    local final_step=$(grep -oP '\[stop\].*?step:\K\d+' "$logfile" | tail -1 || echo "-")
    local peak_alloc=$(grep -oP 'peak_alloc_gib=\K[\d.]+' "$logfile" | tail -1 || echo "-")
    # Average step time from last 5 dt lines
    local avg_dt=$(grep -oP 'dt:\K[\d.]+(?=ms)' "$logfile" | tail -20 | awk '{sum+=$1; n++} END {if(n>0) printf "%.0f", sum/n; else print "-"}')

    local notes=""
    if [ "$avg_dt" != "-" ] && [ "$avg_dt" -lt 400 ] 2>/dev/null; then
        notes="fast"
    elif [ "$avg_dt" != "-" ] && [ "$avg_dt" -gt 550 ] 2>/dev/null; then
        notes="slow"
    fi

    echo ">>> [$label] S200=$s200 S400=$s400 S600=$s600 final=$final_bpb dt=${avg_dt}ms peak=${peak_alloc}gib step=$final_step $notes" | tee -a "$SUMMARY"
    echo "$label | $s200 | $s400 | $s600 | $final_bpb | $avg_dt | $peak_alloc | $final_step | $notes" | tee -a "$SUMMARY"
}

# =============================================================================
# Sweep: 6 modes × 10 min each (~60 min total)
# =============================================================================

# 1. encoder — current default (checkpoint encoder blocks only)
run_checkpoint_test "encoder" 1 encoder

# 2. full — checkpoint ALL blocks (max VRAM save, slight overhead)
run_checkpoint_test "full" 1 full

# 3. decoder — checkpoint decoder blocks only (least VRAM save, fastest)
run_checkpoint_test "decoder" 1 decoder

# 4. alternate — checkpoint every other block (balanced)
run_checkpoint_test "alternate" 1 alternate

# 5. encoder_grouped — checkpoint entire encoder as one group (aggressive VRAM save)
run_checkpoint_test "encoder_grouped" 1 encoder_grouped

# 6. off — no checkpointing (baseline, highest VRAM, most steps)
run_checkpoint_test "off" 0 off

echo "" | tee -a "$SUMMARY"
echo "=== Sweep Complete ===" | tee -a "$SUMMARY"
echo "Results: $SUMMARY" | tee -a "$SUMMARY"
echo "Logs: $LOG_DIR/" | tee -a "$SUMMARY"
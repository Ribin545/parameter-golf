#!/usr/bin/env bash
set -euo pipefail
cd /mnt/e/Projects/Proj/golf/records/track_non_record_16mb/2026-04-01_EliteUTv22p8_12StepRecurrence_Windows3090
source /home/tltp2128/.venvs/pg3090/bin/activate

# Common base
export DATA_PATH=/mnt/e/Projects/Proj/golf/data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=/mnt/e/Projects/Proj/golf/data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export MODEL_TYPE=multilayer
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=2
export RECURRENCE_STEPS=2
export MULTILAYER_LORA_RANK=8
export TRAIN_BATCH_TOKENS=524288
export MICRO_BATCH_TOKENS=153600
export BIGRAM_HASH_ENABLED=1
export BIGRAM_HASH_SIZE=4096
export BIGRAM_HASH_SCALE=0.05
export SHELL_CENTERING_ENABLED=1
export SHELL_CENTERING_LAM=0.005
export MINI_DEPTH_STATIC=1
export MINI_DEPTH_REFINE_BLOCKS=3
export ATTN_OUTPUT_MODE=einsum_fused
export SDPA_BACKEND=flash
export MLP_RECOMPUTE=0
export MLP_MEMORY_MODE=off
export ATTN_MEMORY_MODE=off
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=50
export QUANT_EVAL=1
export QUANT_EVAL_MAX_STEPS=50
export QUANT_EVAL_STRIDE=64
export ITERATIONS=999999
export MAX_WALLCLOCK_SECONDS=600
export DROPOUT_P=0.20
export LABEL_SMOOTHING=0.08
export GRAD_CLIP_NORM=1.0

# ============================================
# E: 6-layer + aggressive LR (combined D+B)
# ============================================
echo "===== RUNNING E: 6-layer + aggrLR ====="
export RUN_ID=ten_min_E_6layer_aggrLR
export NUM_LAYERS=6
export WARMUP_STEPS=40
export MATRIX_LR=0.12
export SCALAR_LR=0.02
export LORA_LR=0.02
export CONTROL_LR=0.02
export MULTILAYER_ACTIVATION_CHECKPOINT=0
export MULTILAYER_ACTIVATION_CHECKPOINT_MODE=off

python3 -u train_gpt.py 2>&1 | tee /tmp/ten_min_E_live.log
echo "--- E RESULT ---"
grep -E '\[stop\]|\[final\]\[quant_eval\]|step:[0-9]+ loss:' /tmp/ten_min_E_live.log | tail -10
echo

# ============================================
# F: 6-layer + encoder_only checkpointing
# ============================================
echo "===== RUNNING F: 6-layer + encoder_only ckpt ====="
export RUN_ID=ten_min_F_6layer_ckpt
export NUM_LAYERS=6
export WARMUP_STEPS=120
export MATRIX_LR=0.08
export SCALAR_LR=0.015
export LORA_LR=0.015
export CONTROL_LR=0.015
export MULTILAYER_ACTIVATION_CHECKPOINT=1
export MULTILAYER_ACTIVATION_CHECKPOINT_MODE=encoder_only

python3 -u train_gpt.py 2>&1 | tee /tmp/ten_min_F_live.log
echo "--- F RESULT ---"
grep -E '\[stop\]|\[final\]\[quant_eval\]|step:[0-9]+ loss:' /tmp/ten_min_F_live.log | tail -10
echo

echo "===== ALL RUNS COMPLETE ====="
#!/usr/bin/env bash
set -euo pipefail
cd /mnt/e/Projects/Proj/golf/records/track_non_record_16mb/2026-04-01_EliteUTv22p8_12StepRecurrence_Windows3090
source /home/tltp2128/.venvs/pg3090/bin/activate

# Best-known 3090 winning config from Phase 11.
# Result: fp_val_bpb=1.586353 in 10 minutes on RTX 3090.

export RUN_ID=best_known_3090_phase11
export DATA_PATH=/mnt/e/Projects/Proj/golf/data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=/mnt/e/Projects/Proj/golf/data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export MODEL_TYPE=multilayer
export NUM_LAYERS=6
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=2
export RECURRENCE_STEPS=2
export MULTILAYER_LORA_RANK=8
export TRAIN_BATCH_TOKENS=524288
export MICRO_BATCH_TOKENS=153600

# Winning speed/quality toggles
export BIGRAM_HASH_ENABLED=0
export SHELL_CENTERING_ENABLED=0
export MINI_DEPTH_STATIC=1
export MINI_DEPTH_REFINE_BLOCKS=2
export RECURRENT_ATTN_EVERY=1
export ATTN_OUTPUT_MODE=baseline
export SDPA_BACKEND=flash
export MULTILAYER_ACTIVATION_CHECKPOINT=0
export MULTILAYER_ACTIVATION_CHECKPOINT_MODE=off
export MLP_RECOMPUTE=0
export MLP_MEMORY_MODE=off
export ATTN_MEMORY_MODE=off
export FUSED_MLP_ENABLED=1
export EMA_UPDATE_EVERY=0

# Evaluation / schedule / objective alignment
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=50
export QUANT_EVAL=1
export QUANT_EVAL_MAX_STEPS=50
export QUANT_EVAL_STRIDE=64
export ITERATIONS=999999
export MAX_WALLCLOCK_SECONDS=600
export WARMUP_STEPS=120
export MATRIX_LR=0.08
export SCALAR_LR=0.015
export LORA_LR=0.015
export CONTROL_LR=0.015
export DROPOUT_P=0.0
export LABEL_SMOOTHING=0.0
export LOGIT_SOFTCAP=30.0
export Z_LOSS_LAMBDA=0.0
export LM_BIAS_INIT=0
export GRAD_CLIP_NORM=1.0
export ACCUM_BACKWARD_SCALE=sum
export OFFICIAL_EVAL_MODE=1

python3 -u train_gpt.py 2>&1 | tee /tmp/best_known_3090_phase11.log
echo "--- BEST KNOWN 3090 RESULT ---"
grep -E '\[stop\]|\[final\]\[quant_eval\]|step:[0-9]+ loss:' /tmp/best_known_3090_phase11.log | tail -30
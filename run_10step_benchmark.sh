#!/bin/bash
cd /mnt/e/Projects/Proj/golf/records/track_non_record_16mb/2026-04-01_EliteUTv22p8_12StepRecurrence_Windows3090
rm -f best_model.pt best_model.int8.ptz

export MLP_MEMORY_MODE=checkpoint
export ATTN_MEMORY_MODE=checkpoint
export SDPA_BACKEND=math
export ATTN_OUTPUT_MODE=baseline
export QK_POST_MODE=baseline
export SAFETY_CLAMP_DISABLE=1
export MODEL_TYPE=multilayer
export NUM_LAYERS=5
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=2
export RECURRENCE_STEPS=2
export MULTILAYER_LORA_RANK=8
export MICRO_BATCH_TOKENS=102400
export TRAIN_BATCH_TOKENS=102400
export TRAIN_SEQ_LEN=1024
export OPTIM_MODE=muon_adam
export MATRIX_OPTIM=muon
export MATRIX_LR=0.12
export SCALAR_LR=0.03
export LORA_LR=0.03
export CONTROL_LR=0.03
export EMBED_LR=0.3
export HEAD_LR=0.008
export TIED_EMBED_LR=0.03
export MUON_BACKEND_STEPS=7
export MUON_MOMENTUM=0.95
export BETA1=0.9
export BETA2=0.95
export SCALAR_WEIGHT_DECAY=0.20
export LORA_WEIGHT_DECAY=0.0
export CONTROL_WEIGHT_DECAY=0.0
export DYNAMIC_LR_NORM=1
export TARGET_GRAD_NORM=0.5
export GRAD_CLIP_NORM=1.0
export DISABLE_COMPILE=1
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
export LOGIT_SOFTCAP=30.0
export QK_GAIN_INIT=3.0
export Z_LOSS_LAMBDA=0.0
export SCHEDULE_FREE=1
export WARMUP_STEPS=20
export TRAIN_LOG_EVERY=1
export VAL_LOSS_EVERY=99999
export SAVE_BEST_CHECKPOINT=0
export QUANT_EVAL=0
export DATA_DETERMINISTIC=1
export DATA_SEED=3623123517
export VOCAB_SIZE=1024
export ITERATIONS=12
export RUN_ID=ph_10step

/home/tltp2128/.venvs/pg3090/bin/python -u train_gpt.py > /tmp/ph_10step.log 2>&1
echo "EXIT_CODE=$?" >> /tmp/ph_10step.log
@echo off
REM B1 Gate: Schedule-Free (constant LR, no warmup/decay) + DYNORM=1 baseline
setlocal
cd /d "%~dp0"
:: Always run via the local venv
if not exist ".\venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Please run .\setup_elite_env.bat first.
    pause
    exit /b 1
)
:: --- B1 toggle ---
set SCHEDULE_FREE=1
:: --- Phase 7 Winner settings ---
set DYNAMIC_LR_NORM=1
set TARGET_GRAD_NORM=0.5
set WARMUP_STEPS=16
set GRAD_CLIP_NORM=1.0
set SHELL_CENTERING_ENABLED=1
set SHELL_CENTERING_LAM=0.008
set BIGRAM_HASH_ENABLED=1
set BIGRAM_HASH_SIZE=2048
set BIGRAM_HASH_SCALE=0.05
set LEVEL_SIGNAL_ENABLED=0
set LORA_SCOPE=q
set LORA_RANK=512
set MLP_MULT=5
set MODEL_DIM=1024
set NUM_HEADS=8
set NUM_KV_HEADS=4
set QK_GAIN_INIT=1.5
set MUON_BACKEND_STEPS=5
set DATA_PATH=..\..\..\data\datasets\fineweb10B_sp1024
set TOKENIZER_PATH=..\..\..\data\tokenizers\fineweb_1024_bpe.model
set VOCAB_SIZE=1024
:: --- Gate test params ---
set ITERATIONS=75
set MAX_WALLCLOCK_SECONDS=600
set VAL_LOSS_EVERY=25
set RUN_ID=b1_schedule_free_gate
set QUANT_EVAL=0
set SAVE_BEST_CHECKPOINT=0
set NO_PAUSE=1
echo =======================================================
echo B1 Gate: SCHEDULE_FREE=1 (75 iters)
echo =======================================================
.\venv\Scripts\python train_gpt_windows.py
endlocal
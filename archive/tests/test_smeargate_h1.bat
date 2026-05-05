@echo off
REM H1: SmearGate WITHOUT ShellCentering (test conflict hypothesis)
setlocal
cd /d "%~dp0"
if not exist ".\venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Please run .\setup_elite_env.bat first.
    pause
    exit /b 1
)
:: --- SmearGate ON, ShellCentering OFF ---
set SMEARGATE_ENABLED=1
set SMEARGATE_ALPHA=0.08
set SHELL_CENTERING_ENABLED=0
:: --- Phase 7 Winner settings (minus ShellCentering) ---
set DYNAMIC_LR_NORM=1
set TARGET_GRAD_NORM=0.5
set WARMUP_STEPS=16
set GRAD_CLIP_NORM=1.0
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
set RUN_ID=smeargate_h1_noshell
set QUANT_EVAL=0
set SAVE_BEST_CHECKPOINT=0
set NO_PAUSE=1
echo =======================================================
echo H1: SmearGate WITHOUT ShellCentering (75 iters)
echo =======================================================
.\venv\Scripts\python train_gpt_windows.py
endlocal
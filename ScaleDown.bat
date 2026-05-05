@echo off
setlocal
cd /d "%~dp0"
:: Elite Universal Transformer: 10-Minute Wallclock Run (ScaleDown)
:: Based on limits_test_10m.bat, but forces key knobs for the next run.

:: Always run via the local venv
if not exist ".\venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Please run .\setup_elite_env.bat first.
    pause
    exit /b 1
)

:: --- Forced run parameters (override any pre-set env vars) ---
set MODEL_DIM=1024

if not defined ITERATIONS set ITERATIONS=415
if not defined MAX_WALLCLOCK_SECONDS set MAX_WALLCLOCK_SECONDS=600
set VAL_LOSS_EVERY=100
set TTT_ENABLED=0
set LOSS_FILTER_ENABLED=1
set MLP_MULT=5
set LEVEL_SIGNAL_ENABLED=0


set LOSS_FILTER_ENABLED=1
set LOSS_FILTER_WARMUP=250
set LOSS_FILTER_MAX_RECENT_DROP=0.02
set LOSS_FILTER_Z_THRESHOLD=4.5

:: Dynamic LR Norm (Phase 7 winner: on + target 0.5, beats baseline by 0.0054 BPB)
if not defined DYNAMIC_LR_NORM set DYNAMIC_LR_NORM=1
if not defined TARGET_GRAD_NORM set TARGET_GRAD_NORM=0.5
if not defined GRAD_CLIP_NORM set GRAD_CLIP_NORM=1.0

:: Shell Centering ENABLED -- lambda overridable via env; default 0.008 (sweep winner: 0.008 > 0.064 > 0.256)
if not defined SHELL_CENTERING_ENABLED set SHELL_CENTERING_ENABLED=1
if not defined SHELL_CENTERING_LAM set SHELL_CENTERING_LAM=0.008

:: Depth-LoRA scope control (new)
:: Values: full | qv | q | none
if not defined LORA_SCOPE set LORA_SCOPE=q
:: --- Defaults / other knobs (leave consistent with limits_test_10m.bat) ---
if not defined WARMUP_STEPS set WARMUP_STEPS=16

if not defined DATA_PATH set DATA_PATH=..\..\..\data\datasets\fineweb10B_sp1024
if not defined TOKENIZER_PATH set TOKENIZER_PATH=..\..\..\data\tokenizers\fineweb_1024_bpe.model
if not defined VOCAB_SIZE set VOCAB_SIZE=1024
if not defined DATA_DETERMINISTIC set DATA_DETERMINISTIC=1
if not defined DATA_SEED set DATA_SEED=3623123517

if not defined NUM_HEADS set NUM_HEADS=8
if not defined NUM_KV_HEADS set NUM_KV_HEADS=4
:: Recurrence depth / reasoning steps
:: Prefer RECURRENCE_STEPS if you want explicit control (it overrides NUM_STEPS).
if not defined RECURRENCE_STEPS if not defined NUM_STEPS set NUM_STEPS=1

set TRAIN_BATCH_TOKENS=524288
set GRAD_ACCUM_STEPS=16
set TRAIN_SEQ_LEN=1024
if not defined LORA_RANK set LORA_RANK=512

if not defined SEQ_LEN_CURRICULUM set SEQ_LEN_CURRICULUM=0
if not defined SEQ_LEN_CURRICULUM_STEPS set SEQ_LEN_CURRICULUM_STEPS=20
if not defined SHORT_TRAIN_SEQ_LEN set SHORT_TRAIN_SEQ_LEN=128

if not defined RECURRENCE_CURRICULUM set RECURRENCE_CURRICULUM=0

if not defined MUON_BACKEND_STEPS set MUON_BACKEND_STEPS=5
:: --- QK Gain Init (WARNING: 5.0 wins at 75-iters but regresses at 300+ iters; keep default 1.5) ---
if not defined QK_GAIN_INIT set QK_GAIN_INIT=1.5
:: --- Learning Rates (exposed; default to trainer defaults unless overridden) ---
if not defined MATRIX_LR set MATRIX_LR=0.08
if not defined SCALAR_LR set SCALAR_LR=0.015
if not defined LORA_LR set LORA_LR=%SCALAR_LR%
if not defined CONTROL_LR set CONTROL_LR=%SCALAR_LR%
if not defined SCALAR_WEIGHT_DECAY set SCALAR_WEIGHT_DECAY=0.1
if not defined LORA_WEIGHT_DECAY set LORA_WEIGHT_DECAY=0.0
if not defined CONTROL_WEIGHT_DECAY set CONTROL_WEIGHT_DECAY=0.0
if not defined EMBED_LR set EMBED_LR=0.3
if not defined HEAD_LR set HEAD_LR=0.008
if not defined TIED_EMBED_LR set TIED_EMBED_LR=0.06
:: --- BigramHash feature toggle ---
if not defined BIGRAM_HASH_ENABLED set BIGRAM_HASH_ENABLED=1
if not defined BIGRAM_HASH_SIZE set BIGRAM_HASH_SIZE=2048
if not defined BIGRAM_HASH_SCALE set BIGRAM_HASH_SCALE=0.05

echo =======================================================
echo Elite Universal Transformer: ScaleDown Run
echo Standard: 524k Global Batch ^| 12-Step Reasoning
echo Forced: MODEL_DIM=%MODEL_DIM% ^| MLP_MULT=%MLP_MULT% ^| ITERS=%ITERATIONS%
echo Filter: LOSS_FILTER_ENABLED=%LOSS_FILTER_ENABLED%
echo LRs: MATRIX_LR=%MATRIX_LR% ^| SCALAR_LR=%SCALAR_LR% ^| EMBED_LR=%EMBED_LR% ^| HEAD_LR=%HEAD_LR% ^| TIED_EMBED_LR=%TIED_EMBED_LR%
echo Optim Groups: LORA_LR=%LORA_LR% ^| CONTROL_LR=%CONTROL_LR% ^| SCALAR_WD=%SCALAR_WEIGHT_DECAY% ^| LORA_WD=%LORA_WEIGHT_DECAY% ^| CONTROL_WD=%CONTROL_WEIGHT_DECAY%
echo Grad: DYNAMIC_LR_NORM=%DYNAMIC_LR_NORM% ^| TARGET_GRAD_NORM=%TARGET_GRAD_NORM% ^| GRAD_CLIP_NORM=%GRAD_CLIP_NORM%
echo Features: BIGRAM_HASH_ENABLED=%BIGRAM_HASH_ENABLED% ^| BIGRAM_HASH_SIZE=%BIGRAM_HASH_SIZE% ^| BIGRAM_HASH_SCALE=%BIGRAM_HASH_SCALE% ^| LEVEL_SIGNAL_ENABLED=%LEVEL_SIGNAL_ENABLED% ^| SHELL_CENTERING=%SHELL_CENTERING_ENABLED% ^| SHELL_CENTERING_LAM=%SHELL_CENTERING_LAM%
echo Data Seed: DATA_DETERMINISTIC=%DATA_DETERMINISTIC% ^| DATA_SEED=%DATA_SEED%
echo Limit: %MAX_WALLCLOCK_SECONDS% Seconds
echo =======================================================

.\venv\Scripts\python train_gpt_windows.py

echo.
echo =======================================================
echo Training Period Complete. Checking model.pt for BPB.
echo =======================================================
if "%NO_PAUSE%"=="1" (
    echo [bat] NO_PAUSE=1, exiting without pause.
) else (
    pause
)

@echo off
REM ============================================================================
REM AB Test Runner - Phase 1: Per-Step Quality Gate
REM Uses train_gpt_windows.py infra via MODEL_TYPE env var.
REM Expected: ~2 min per config (~4 min total per methodology spec).
REM ============================================================================
cd /d "%~dp0"

set DATA_PATH=e:\Projects\Proj\golf\data\datasets\fineweb10B_sp1024
set TOKENIZER_PATH=e:\Projects\Proj\golf\data\tokenizers\fineweb_1024_bpe.model
set PYTHONUNBUFFERED=1

echo.
echo ===========================================================================
echo AB TEST - Phase 1: Per-Step Quality Gate
echo Started at %date% %time%
echo ===========================================================================
echo.

python -u ab_test.py 2>&1

echo.
echo ===========================================================================
echo AB TEST COMPLETE at %date% %time%
echo Logs saved to ab_logs\config_a.txt and ab_logs\config_b.txt
echo ===========================================================================
pause
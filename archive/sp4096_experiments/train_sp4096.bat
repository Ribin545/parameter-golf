@echo off
setlocal
set LOG=e:\Projects\Proj\golf\records\track_non_record_16mb\2026-04-01_EliteUTv22p8_12StepRecurrence_Windows3090\logs\p8_A1_SP4096_full_v5.log
echo [%date% %time%] === SP4096: Downloading docs_selected.jsonl via huggingface-cli === > %LOG%
cd /d "e:\Projects\Proj\golf\data"
hf download willdepueoai/parameter-golf datasets/docs_selected.jsonl --repo-type dataset --local-dir . >> %LOG% 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [%date% %time%] ERROR: Download failed with code %ERRORLEVEL% >> %LOG%
  exit /b %ERRORLEVEL%
)
echo [%date% %time%] === SP4096: Download complete. Starting tokenizer training + tokenization === >> %LOG%
python download_hf_docs_and_tokenize.py --output-root . >> %LOG% 2>&1
echo [%date% %time%] === SP4096: Pipeline complete (exit code %ERRORLEVEL%) === >> %LOG%

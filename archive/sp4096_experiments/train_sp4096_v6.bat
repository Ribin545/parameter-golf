@echo off
setlocal
set LOG=e:\Projects\Proj\golf\records\track_non_record_16mb\2026-04-01_EliteUTv22p8_12StepRecurrence_Windows3090\logs\p8_A1_SP4096_v6.log
set WORK_DIR=C:\sp4096_temp
set DEST_DIR=e:\Projects\Proj\golf\data

echo [%date% %time%] === SP4096: Creating work dir on C: === > %LOG%
if exist %WORK_DIR% rmdir /s /q %WORK_DIR%
mkdir %WORK_DIR%
cd /d %WORK_DIR%

echo [%date% %time%] === SP4096: Downloading docs_selected.jsonl to C: (115GB available) === >> %LOG%
hf download willdepueoai/parameter-golf datasets/docs_selected.jsonl --repo-type dataset --local-dir %WORK_DIR% >> %LOG% 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [%date% %time%] ERROR: Download failed with code %ERRORLEVEL% >> %LOG%
  exit /b %ERRORLEVEL%
)

echo [%date% %time%] === SP4096: Copying tokenizer config to work dir === >> %LOG%
copy %DEST_DIR%\tokenizer_specs.json %WORK_DIR%\ >nul 2>&1

echo [%date% %time%] === SP4096: Training tokenizer + tokenizing (this may take 10-20 min) === >> %LOG%
python %DEST_DIR%\download_hf_docs_and_tokenize.py --output-root %WORK_DIR% >> %LOG% 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [%date% %time%] ERROR: Tokenization failed with code %ERRORLEVEL% >> %LOG%
  exit /b %ERRORLEVEL%
)

echo [%date% %time%] === SP4096: Moving tokenized shards to E: === >> %LOG%
if exist %WORK_DIR%\fineweb10B_sp4096 (
  if exist %DEST_DIR%\fineweb10B_sp4096 rmdir /s /q %DEST_DIR%\fineweb10B_sp4096
  move %WORK_DIR%\fineweb10B_sp4096 %DEST_DIR%\fineweb10B_sp4096 >> %LOG% 2>&1
)

echo [%date% %time%] === SP4096: Copying tokenizer files to E: === >> %LOG%
if exist %WORK_DIR%\fineweb_4096_bpe.model copy %WORK_DIR%\fineweb_4096_bpe.model %DEST_DIR%\ >nul 2>&1
if exist %WORK_DIR%\fineweb_4096_bpe.vocab copy %WORK_DIR%\fineweb_4096_bpe.vocab %DEST_DIR%\ >nul 2>&1

echo [%date% %time%] === SP4096: Cleaning up C: work dir === >> %LOG%
rmdir /s /q %WORK_DIR%

echo [%date% %time%] === SP4096: Pipeline complete! === >> %LOG%
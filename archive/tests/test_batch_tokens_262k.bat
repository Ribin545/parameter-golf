@echo off
setlocal
cd /d "%~dp0"

set TRAIN_BATCH_TOKENS=262144
set GRAD_ACCUM_STEPS=8
set NO_PAUSE=1

call ScaleDown.bat
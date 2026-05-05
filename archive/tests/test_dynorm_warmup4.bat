@echo off
setlocal
cd /d "%~dp0"

set DYNAMIC_LR_NORM=1
set TARGET_GRAD_NORM=0.5
set WARMUP_STEPS=4
set NO_PAUSE=1

call ScaleDown.bat
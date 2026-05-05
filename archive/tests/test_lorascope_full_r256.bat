@echo off
setlocal
cd /d "%~dp0"

set LORA_SCOPE=full
set LORA_RANK=256
set NO_PAUSE=1

call ScaleDown.bat
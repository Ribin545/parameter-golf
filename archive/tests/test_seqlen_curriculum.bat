@echo off
setlocal
cd /d "%~dp0"

set SEQ_LEN_CURRICULUM=1
set SEQ_LEN_CURRICULUM_STEPS=20
set SHORT_TRAIN_SEQ_LEN=128
set NO_PAUSE=1

call ScaleDown.bat
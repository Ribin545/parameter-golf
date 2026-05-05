@echo off
setlocal
cd /d "%~dp0"

:: Phase 7 Extension: ShellCentering λ=0.256 × 20 min
:: Hypothesis: λ=0.256 had best early learning (Phase 4) but degraded late (Phase 5).
:: With DYNAMIC_LR_NORM=1, the late-training gradient control may prevent degradation.
set SHELL_CENTERING_LAM=0.256
set MAX_WALLCLOCK_SECONDS=1200
set ITERATIONS=830
set NO_PAUSE=1

call ScaleDown.bat
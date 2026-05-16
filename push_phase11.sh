#!/usr/bin/env bash
set -euo pipefail
cd /mnt/e/Projects/Proj/golf/records/track_non_record_16mb/2026-04-01_EliteUTv22p8_12StepRecurrence_Windows3090
git add -A
git status --short
git commit -m 'phase11: gap closed from +0.051 to +0.011 via 6 config fixes (G_clean_aligned=1.6154). Added ACCUM_BACKWARD_SCALE, OFFICIAL_EVAL_MODE. Updated report.'
/mnt/c/Program\ Files/Git/cmd/git.exe push origin master
echo 'EXIT=True'

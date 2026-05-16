#!/usr/bin/env bash
set -euo pipefail
cd /mnt/e/Projects/Proj/golf/records/track_non_record_16mb/2026-04-01_EliteUTv22p8_12StepRecurrence_Windows3090
git add -A
git status --short
git commit -m 'phase11: 3090 524k gap analysis + 10min sweep results (B/D/E/F configs)'
/mnt/c/Program\ Files/Git/cmd/git.exe push origin master
echo 'EXIT=True'
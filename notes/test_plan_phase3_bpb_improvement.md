# Phase 3: BPB Improvement Plan — Target <1.4 BPB under 600ms

## Context
C7 baseline: dim=512, mlp_mult=6, recur=1, compile=ON → 581ms, best 1.8503 BPB @400
Problem: Model plateaus at step 400 (233s), zero improvement for remaining 367s.
Target: 200-step gate test (val_bpb <2.6), then 10-minute run.

## Screening Protocol
- MAX_WALLCLOCK_SECONDS=9999 (unlimited for 200-step gate)
- ITERATIONS=200, VAL_LOSS_EVERY=100
- Gate pass: val_bpb < 2.6 at step 200 AND step_time < 600ms
- Winner: best val_bpb under 600ms

---

## Track 1: LR Dynamics Fix (C7 chassis, zero speed cost)
Goal: Keep model improving past step 400 by fixing LR schedule.

| ID | Dim | MLP_MULT | Recur | MATRIX_LR | EMBED_LR | HEAD_LR | WARMUP | Est DT | Notes |
|----|----:|-----|-----|-----|-----|-----|-----|-----:|-------|
| **T1A** | 512 | 6 | 1 | 0.12 | 0.4 | 0.012 | 2 | ~581ms | Higher LR ceiling |
| **T1B** | 512 | 6 | 1 | 0.16 | 0.5 | 0.016 | 2 | ~581ms | Aggressive LR |
| **T1C** | 512 | 6 | 1 | 0.10 | 0.35 | 0.010 | 1 | ~581ms | Moderate LR, fast warmup |

---

## Track 2: Recurrence=2 on Diet Chassis
Goal: Depth > width. C4 single-step is 493ms. Recur=2 ~1.6x cost.

| ID | Dim | MLP_MULT | Recur | Est DT | Head Dim | Notes |
|----|----:|-----|-----|-----:|-----:|-------|
| **T2A** | 384 | 4 | 2 | ~540ms | 96 | Mid recur |
| **T2B** | 320 | 4 | 2 | ~470ms | 80 | Light recur |
| **T2C** | 384 | 3 | 2 | ~480ms | 96 | Lean MLP recur |
| **T2D** | 256 | 4 | 2 | ~420ms | 64 | Very light |

---

## Track 3: Width Exploration (recur=1)
Goal: More capacity per step.

| ID | Dim | MLP_MULT | Recur | Est DT | Head Dim | Notes |
|----|----:|-----|-----|-----:|-----:|-------|
| **T3A** | 640 | 3 | 1 | ~580ms | 160 | Wide, lean MLP |
| **T3B** | 576 | 4 | 1 | ~620ms | 144 | Wide, mid MLP (over budget) |
| **T3C** | 576 | 3 | 1 | ~530ms | 144 | Wide, lean |

---

## Execution Order
1. T1A, T1B, T1C (fastest to test, zero chassis change)
2. T2A, T2B, T2C, T2D (recurrence exploration)
3. T3A, T3C (width exploration, T3B likely over budget)

Each test: 200 iterations (~2min). Check val_bpb at step 100/200.
If any passes (<2.6 BPB, <600ms), flag for 10-minute run.
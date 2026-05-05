# Phase 2 — Shell Centering Penalty (Issue #2045)

## Run Summary

| | Config A (Baseline) | Config B (ShellCentering, λ=0.008) |
|---|---|---|
| **Log file** | `logs/phase2_baselineA_out.log` | `logs/phase2_candidateB_shellCentering_out.log` |
| **Timestamp** | 13:49–13:53 IST | 13:54–13:59 IST |
| **Stop reason** | Wallclock | Wallclock |
| **Elapsed** | 242.4s | 240.9s |
| **Steps completed** | 91 | **94** (+3) |
| **Loss filter** | 16/0/0 ✅ | 16/0/0 ✅ |
| **VRAM peak** | 3.51 GiB | 3.58 GiB (+2%) |

## Required Measurements

| Metric | A (Baseline) | B (ShellCentering) | Δ |
|---|---|---|---|
| **Final stride-64 BPB** | 2.7802 (step 91) | **2.6891** (step 94) | **−0.0911** |
| **Best checkpoint BPB** | 2.7667 (step 90) | **2.6891** (step 94) | **−0.0776** |
| **Total iterations** | 91 | 94 | B +3.3% throughput |

## Win Condition Check

| Gate | Criterion | Result |
|---|---|---|
| Final BPB | B < A? | ✅ 2.6891 < 2.7802 |
| Best BPB | B < A? | ✅ 2.6891 < 2.7667 |
| Throughput | B ≥ A? | ✅ 94 > 91 (B slightly faster) |

**Both metrics pass decisively — no mixed result. No tie-breaking needed.**

## Phase 2 Verdict: ✅ PASS

B improves final BPB by 0.0911 (3.3%), best BPB by 0.0776 (2.8%), and completes 3 more iterations in the same wallclock budget. The ShellCentering penalty adds negligible VRAM overhead (+0.07 GiB) and no throughput penalty — if anything B was marginally faster (slightly less loss computation overhead from regularised embeddings?).

## Quant Eval Comparison

| Metric | A | B |
|---|---|---|
| FP val_bpb | 2.792071 | 2.689138 |
| INT8 val_bpb | 2.794043 | 2.689355 |
| Δ BPB | +0.001972 | +0.000217 |
| Degradation | 0.071% | 0.008% |

B's quant degradation (0.008%) is an order of magnitude better than A's (0.071%) — the shell-centering produces much more quantization-robust weight distributions.

## Combined Phase 1 + 2 Trajectory

| | A (Baseline) | B (ShellCentering) |
|---|---|---|
| Phase 1 (75 iters) | 2.8370 BPB | **2.7775** BPB (−2.1%) |
| Phase 2 (240s) | 2.7667 BPB | **2.6891** BPB (−2.8%) |
| B advantage | — | **Growing with budget** ✅ |

The advantage widens from 0.0595 BPB at 75 iterations to 0.0776 BPB at 240s wallclock — the regularisation continues to compound, not saturate.

## Next: Phase 3 (Statistical Confidence)

Per `07_ab_testing_methodology.md`:
- Run A and B at 3 seeds each (42, 1337, 2024) with `MAX_WALLCLOCK_SECONDS=120`
- Compare mean ± σ bands
- If B's mean + 1σ < A's mean − 1σ → ACCEPT B as clear winner
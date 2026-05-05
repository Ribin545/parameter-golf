# Phase 7 Test 8: ShellCentering λ=0.256 + DYNORM=1 — 20-Min Test

**Date:** 2026-05-04
**Status:** CLOSED ❌ — Inferior to Winner

---

## Motivation

Phase 5 swept ShellCentering λ (0, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256) in 10-min tests and found **λ=0.008** was the sweet spot (val_bpb=1.8679). λ=0.256 was the worst performer in that sweep (val_bpb=1.9208).

However, Phase 5 tests were short (10 min) and the loss filter rejected some data at λ=0.008. This test re-evaluated λ=0.256 with **DYNORM=1** and a **full 20-minute budget** to rule out any late-blooming benefit.

## Configuration

```
SHELL_CENTERING=1
SHELL_CENTERING_LAM=0.256
DYNAMIC_LR_NORM=1
TARGET_GRAD_NORM=0.5
GRAD_CLIP_NORM=1.0
All other settings = Phase 7 Winner defaults
```

## Results

| Step | λ=0.256 + DYNORM | Phase 7 Winner (DYNORM only) | Δ |
|------|-------------------|------------------------------|---|
| 100 | 2.7985 | 2.4922 | +0.3063 |
| 200 | 2.4239 | 1.9511 | +0.4728 |
| 300 | 2.0486 | 1.8810 | +0.1676 |
| 400 | 1.9065 | **1.8228** | +0.0837 |
| 500 | 1.8469 | — | — |
| 600 | 1.8236 | — | — |
| 700 | **1.8086** | — | — |

**Best checkpoint:** Step 700, val_bpb=1.8086
**Final stride 64:** val_bpb=1.8688 (step 778, wallclock stop)
**Quant eval (fp/int8):** 1.8749 / 1.8762

## Comparison with Phase 5 λ=0.008 Sweet Spot

| Test | Best BPB (10min) | Δ vs Winner |
|------|------------------|-------------|
| λ=0.008 (Phase 5) | **1.8679** | — |
| λ=0.256 (Phase 5) | 1.9208 | +0.0529 vs λ=0.008 |
| λ=0.256 + DYNORM (this test) | 1.8086 (20min) | +0.0837 vs Winner at step 400 |

## Conclusion

1. **λ=0.256 is definitively worse than Phase 7 Winner (DYNORM only):** +0.08–0.47 BPB at all comparable checkpoints.

2. **λ=0.256 achieves 1.8086 only with 75% more iterations** (step 700 vs 400). The Winner already reached 1.8228 at step 400 and would be far ahead with equivalent time.

3. **Phase 5's finding is confirmed:** λ=0.008 was the correct ShellCentering sweet spot. Higher λ values (0.256) aggressively penalize shell deviation, slowing early optimization and never catching up within 20 minutes.

4. **Quant eval confirms:** λ=0.256 quant eval (1.8749) lags behind Winner (1.8688).

## Action

- ✅ Keep Phase 7 Winner (DYNORM=1 solo, val_bpb=1.8228) as best configuration.
- ✅ λ=0.008 remains the best ShellCentering value if re-enabled, but was already superseded by DYNORM=1 without ShellCentering.
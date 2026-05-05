# Phase 7, Test 4: MUON_MOMENTUM=0.98

**Date:** 2026-05-03  
**Hypothesis:** Higher momentum (0.98 vs 0.95) → faster accumulation of consistent gradient directions over ~400 steps → better convergence  
**Result:** ❌ **REGRESSION** (best=1.8967 vs DYNAMIC_LR_NORM best=1.8228, +0.0739 worse)

---

## Results vs Comparison

| Step | MUON_MOM=0.98 | DYNAMIC_LR_NORM=1 | Baseline | Delta vs DynLR |
|------|--------------|-------------------|----------|----------------|
| 100 | 2.4867 | 2.4922 | 2.5121 | -0.0055 better |
| 200 | 1.9469 | 1.9511 | 1.9527 | -0.0042 better |
| 300 | 1.8967 | 1.8444 | 1.8500 | +0.0523 worse |
| 400 | — (only 388 steps) | 1.8228 | 1.8282 | N/A |

| Final FP bpb | 1.9384 | 1.8688 | 1.8814 | +0.0696 worse |
| Steps | 388 | 410 | 408 | -22 fewer |
| Speed | ~1580ms | ~1400ms | ~1410ms | ~200ms slower |

---

## Analysis

MUON_MOMENTUM=0.98 was slightly ahead at steps 100 and 200, but the momentum accumulation started hurting after step 250. The 200ms/step speed penalty also cost 22 gradient steps. The model never recovered and the best checkpoint was at step 300 (1.8967), which is far behind the competition.

The higher momentum appears to create an optimizer "over-commitment" problem: past gradients dominate too heavily, preventing course corrections in later training.

---

## Conclusion

Higher Muon momentum hurts both throughput and per-step quality in the critical later training phase. Keeping DYNAMIC_LR_NORM=1 as the current best (1.8228). Next candidate: reduce dropout to let the model learn faster.
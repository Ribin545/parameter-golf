# Phase 7, Test 5: DROPOUT=0.05

**Date:** 2026-05-03  
**Hypothesis:** Reducing dropout from 0.15 to 0.05 → less regularization → faster learning for underfitting model  
**Result:** ❌ **REGRESSION** (best=1.8503 vs DYNAMIC_LR_NORM=1.8228, +0.0275 worse)

---

## Results vs Comparison

| Step | DROPOUT=0.05 | DYNAMIC_LR_NORM=1 | Delta |
|------|-------------|-------------------|-------|
| 100 | 2.5169 | 2.4922 | +0.0247 worse |
| 200 | 1.9612 | 1.9511 | +0.0101 worse |
| 300 | 1.8503 | 1.8444 | +0.0059 worse |
| 400 | — (only 369 steps) | 1.8228 | N/A |

| Final FP bpb | 1.9010 | 1.8688 | +0.0322 worse |
| Steps | 369 | 410 | -41 fewer |
| Speed | ~1570ms | ~1400ms | ~170ms slower |
| Loss filter | 44 skipped (vs 0) | 0 skipped | ⚠️ filter divergence |

---

## Conclusion

Lower dropout hurt convergence at every checkpoint. Interestingly, it also slowed down (1570ms vs 1400ms normal) — likely GPU thermal throttling after consecutive runs. Keeping DYNAMIC_LR_NORM=1 as best (1.8228).
# Phase 7, Test 6: LABEL_SMOOTHING=0.0

**Date:** 2026-05-03  
**Hypothesis:** Removing label smoothing → sharper training signal → faster convergence  
**Result:** ❌ **REGRESSION** (best=1.8543 vs DYNAMIC_LR_NORM=1.8228, +0.0315 worse)

---

## Results vs Comparison

| Step | LABEL_SMOOTH=0.0 | DYNAMIC_LR_NORM=1 | Delta |
|------|-----------------|-------------------|-------|
| 100 | 2.5152 | 2.4922 | +0.0230 worse |
| 200 | 1.9588 | 1.9511 | +0.0077 worse |
| 300 | 1.8543 | 1.8444 | +0.0099 worse |
| 400 | — (only 362 steps) | 1.8228 | N/A |

| Final FP bpb | 1.9031 | 1.8688 | +0.0343 worse |
| Steps | 362 | 410 | -48 fewer |
| Loss filter | 0 skipped | 0 skipped | clean |

---

## Conclusion

Tier 1 fully tested — only DYNAMIC_LR_NORM=1 wins. Moving to Tier 2 candidates with Phase 1 gating.
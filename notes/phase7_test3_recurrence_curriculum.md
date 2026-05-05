# Phase 7, Test 3: RECURRENCE_CURRICULUM=1 (warmup from short recurrence)

**Date:** 2026-05-03  
**Hypothesis:** Warmup from short recurrence steps → better initialization of recurrence path → better convergence  
**Result:** ❌ **REGRESSION** (+0.0076 bpb at best)

---

## Results vs Baseline

| Step | RECURRENCE_CURRICULUM=1 | DYNAMIC_LR_NORM=1 | Delta vs DynLR |
|------|--------------------------|-------------------|----------------|
| 100 | 2.5248 | 2.4922 | +0.0326 worse |
| 200 | 1.9681 | 1.9511 | +0.0170 worse |
| 300 | 1.8592 | 1.8444 | +0.0148 worse |
| 400 | 1.8358 | 1.8228 | +0.0130 worse |

| Final FP bpb | 1.8914 | 1.8688 | +0.0226 worse |
| Steps | 412 | 410 | +2 |
| Speed | ~1420ms | ~1400ms | slightly slower |
| Loss filter | 128 skipped (vs 0) | 0 skipped | ⚠️ filter divergence |

---

## Conclusion

Recurrence curriculum actually hurt — 128 loss filter skips indicate gradient instability during the recurrence ramp phase. The current 12-step recurrence works better with full recurrence from the start. Keeping DYNAMIC_LR_NORM as the current best.
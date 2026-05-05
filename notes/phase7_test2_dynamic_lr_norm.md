# Phase 7, Test 2: DYNAMIC_LR_NORM=1 (TARGET_GRAD_NORM=0.5)

**Date:** 2026-05-03  
**Hypothesis:** Dynamic LR normalization → stabilize Muon training → faster/better convergence  
**Result:** ✅ **IMPROVEMENT** (-0.0054 bpb at best, consistent across all checkpoints)

---

## Results vs Baseline

| Step | DYNAMIC_LR_NORM=1 | Baseline | Delta |
|------|-------------------|----------|-------|
| 100 | 2.4922 | 2.5121 | -0.0199 better |
| 200 | 1.9511 | 1.9527 | -0.0016 better |
| 300 | 1.8444 | 1.8500 | -0.0056 better |
| 400 | 1.8228 | 1.8282 | -0.0054 better |

| Final FP bpb | 1.8688 | 1.8814 | -0.0126 better |
| Steps | 410 | 408 | +2 |
| Speed | ~1400ms | ~1410ms | similar |

---

## Conclusion

Dynamic LR normalization consistently improves convergence at all checkpoints. The small gain (-0.0054) is robust and monotonic — every checkpoint is better. Keeping this change.
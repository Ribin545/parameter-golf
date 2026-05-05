# Phase 7, Test 1: ROPE_BASE = 1,000,000

**Date:** 2026-05-03  
**Hypothesis:** Larger rope base → better gradient flow for Q-LoRA → faster learning  
**Result:** ❌ **REGRESSION** (+0.0139 bpb)

---

## Results vs Baseline

| Step | ROPE_BASE=1M | Baseline (10K) | Delta |
|------|-------------|----------------|-------|
| 100 | 2.5180 | 2.5121 | +0.0059 worse |
| 200 | 1.9619 | 1.9527 | +0.0092 worse |
| 300 | 1.8642 | 1.8500 | +0.0142 worse |
| 400 | 1.8421 | 1.8282 | +0.0139 worse |

| Final FP bpb | 1.9012 | 1.8814 | +0.0198 worse |
| Steps | 412 | 408 | +4 |
| Speed | ~1400ms | ~1410ms | similar |

---

## Conclusion

Larger rope base degrades performance monotonically on this 1024-dim, 1-step model. The GPT-2 era default of 10,000 is well-tuned for the existing architecture. Reverting.
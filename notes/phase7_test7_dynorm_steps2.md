# Phase 7, Test 7: DYNAMIC_LR_NORM=1 + NUM_STEPS=2

**Date:** 2026-05-03  
**Hypothesis:** Combining winning optimizer change (dynamic LR norm) with 2-pass recurrence → deeper reasoning per step, better BPB  
**Result:** ❌ **CATASTROPHIC REGRESSION** (best=2.0607 vs DYNAMIC_LR_NORM=1.8228, +0.2379)

---

## Results vs Comparison

| Step | DYNORM+STEPS=2 | DYNAMIC_LR_NORM=1 | Delta |
|------|---------------|-------------------|-------|
| 100 | 2.4949 | 2.4922 | +0.0027 worse |
| 200 | 2.0607 | 1.9511 | +0.1096 worse |
| 400 | — (only 199 steps) | 1.8228 | N/A |

| Metric | Value |
|--------|-------|
| Steps | 199 (vs 410) |
| Speed | ~2950ms (vs ~1400ms) — 2.1× slower |
| VRAM peak | 5.99 GiB (vs 3.54) |
| LoRA params | 2,097,152 (0.1111 vs 0.0588) |
| Model size | 36.01 MiB (vs 34.01) |
| Loss filter | 0 skipped |

---

## Conclusion

2-pass recurrence is a FAILURE at this budget. The 2.1× compute cost per step consumed the entire 10-min budget with only 199 gradient steps. Even though step-by-step losses were comparable (~3.58 at step 187 vs ~3.48), the model needed 200+ more gradient updates to converge. 

This confirms: **the system is gradient-step-limited, not forward-pass-depth-limited.** More compute per step cannot compensate for fewer steps at this budget.

Remaining viable candidates: WARMUP_STEPS sweep, GRAD_CLIP_NORM sweep, TRAIN_BATCH_TOKENS halved.
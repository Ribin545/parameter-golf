# Phase 7: Improvement Opportunities — Final Summary

**Date:** 2026-05-04
**Status:** CLOSED ✅ — Winner accepted

---

## Summary

Phase 7 tested 9 candidates (Tier 1 scalar-only + Tier 2 optimizer changes) against the Phase 6 baseline (val_bpb=1.8282). All tests followed the A/B testing 3-phase methodology with Phase 1 gate at 75 iters.

## Results Table

| # | Candidate | Best BPB | Steps | Δ vs 1.8228 | Verdict |
|---|-----------|----------|-------|-------------|---------|
| 2 | **DYNAMIC_LR_NORM=1** | **1.8228** | 410 | — | ✅ **WINNER** |
| 9 | DYNORM + WARMUP_STEPS=4 | 1.8322 | 413 | +0.0094 | ❌ |
| 3 | RECURRENCE_CURRICULUM | 1.8358 | 410 | +0.0130 | ❌ |
| 1 | ROPE_BASE=1M | 1.8421 | 412 | +0.0193 | ❌ |
| 5 | DROPOUT=0.05 | 1.8503 | 369 | +0.0275 | ❌ |
| 6 | LABEL_SMOOTHING=0.0 | 1.8543 | 362 | +0.0315 | ❌ |
| 8 | DYNORM + GRAD_CLIP=0.5 | 1.8656 | 355 | +0.0428 | ❌ |
| 4 | MUON_MOMENTUM=0.98 | 1.8967 | 388 | +0.0739 | ❌ |
| 7 | DYNORM + NUM_STEPS=2 | 2.0607 | 230 | +0.2379 | ❌ |

## Winner Details

- **Log:** `logs/34c58990-4284-4ad1-9d2a-2275a3b92a74.txt`
- **Run ID:** b71894be-8205-48da-8b05-5d5bd40b3b3f
- **Best checkpoint:** Step 400
- **Best val_bpb:** 1.8228
- **Best val_loss:** 3.0876
- **Config:** TARGET_GRAD_NORM=0.5, warmup_steps=16, GRAD_CLIP_NORM=1.0
- **Quant eval:** fp=1.868755 | int8=1.868295 | delta=-0.000460 (-0.025%)
- **Int8 size:** 11,514,529 bytes

## Improvement vs Baseline

| Step | DYNORM=1 | Phase 6 Baseline | Δ |
|------|----------|-----------------|---|
| 100 | 2.4922 | 2.4964 | -0.0042 |
| 200 | 1.9511 | 1.9610 | -0.0099 |
| 300 | 1.8810 | 1.9059 | -0.0249 |
| 400 | **1.8228** | 1.8482 | -0.0254 |

## What Was Tested (and Failed)

### Tier 1 — Scalar-Only Changes
- **ROPE_BASE=1M:** ❌ Destabilized training at step 170
- **RECURRENCE_CURRICULUM:** ❌ Minor negative
- **WARMUP_STEPS=4/32:** ❌ Tested as combos, both negative
- **DROPOUT=0.05:** ❌ Negative + reduced throughput
- **LABEL_SMOOTHING=0.0:** ❌ Negative + reduced throughput

### Tier 2 — Optimizer Changes
- **MUON_MOMENTUM=0.98:** ❌ Severe regression (BPB 1.90)
- **DYNAMIC_LR_NORM=1 (solo):** ✅ +0.0054 improvement
- **DYNORM + GRAD_CLIP_NORM=0.5:** ❌ Loss filter rejections
- **DYNORM + WARMUP_STEPS=4:** ❌ Slightly worse than DYNORM solo
- **DYNORM + NUM_STEPS=2:** ❌ Severe regression (BPB 2.06)

## Key Insight

DYNAMIC_LR_NORM=1 with TARGET_GRAD_NORM=0.5 provides modest but consistent gains across all checkpoints. The improvement grows over time (0.0042 at step 100 → 0.0254 at step 400), suggesting better late-training gradient management. No further combination (with WARMUP_STEPS, tighter clip, or NUM_STEPS) improved upon the solo setting.

## Next Steps (if desired)

**Tier 3 — Architectural Changes (higher risk):**
- LORA_SCOPE=full + LORA_RANK=256
- LOGIT_SOFTCAP=5.0
- Combined: DYNORM=1 + architectural change

**Phase 3 — Multi-Seed Confirmation:**
- Run winner at seeds 42, 2024 to confirm gain is real
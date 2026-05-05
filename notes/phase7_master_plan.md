# Phase 7 Master Plan & Tally

## Current Leaderboard (All Tests)

| Test | Candidate | Best BPB | Δ vs Best | Steps | Verdict |
|------|-----------|----------|-----------|-------|---------|
| 2 | DYNAMIC_LR_NORM=1 | **1.8228** | — (leader) | 410 | ✅ ACCEPT |
| Baseline | Phase 6 | 1.8282 | +0.0054 | 408 | — |
| 3 | RECURRENCE_CURRICULUM | 1.8358 | +0.0130 | 410 | ❌ REJECT |
| 1 | ROPE_BASE=1M | 1.8421 | +0.0193 | 412 | ❌ REJECT |
| 6 | LABEL_SMOOTHING=0.0 | 1.8543 | +0.0315 | 362 | ❌ REJECT |
| 5 | DROPOUT=0.05 | 1.8503 | +0.0275 | 369 | ❌ REJECT |
| 4 | MUON_MOMENTUM=0.98 | 1.8967 | +0.0739 | 388 | ❌ REJECT |

## Tier 1 Status: FULLY EXHAUSTED
Only winner: DYNAMIC_LR_NORM=1

## Next: Tier 2 — Build on Winner
**Test 7: DYNAMIC_LR_NORM=1 + NUM_STEPS=2 (deeper recurrence)**
- Hypothesis: 2-pass recurrence per forward step gives model more reasoning depth
- Trade-off: ~200 gradient steps instead of ~400, but each step has 2× block computation
- Method: Full 10-min run (600s), compare BPB vs DYNAMIC_LR_NORM=1 baseline

**After Test 7 — remaining Tier 2 candidates (in order):**
- WARMUP_STEPS sweep (4 or 32)
- TRAIN_BATCH_TOKENS halved (262144)
- GRAD_CLIP_NORM sweep (0.5 or 2.0)

## Combination to try if individual Tier 2 wins:
DYNAMIC_LR_NORM=1 + [best Tier 2 winner] + [best Tier 3 speculative]
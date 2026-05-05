# Phase 7: Improvement Opportunities — Architecture & Hyperparameter Ideas

**Date:** 2026-05-03  
**Purpose:** Identify untested combinations that could improve val_bpb beyond 1.8282

---

## Current Baseline (Phase 6)

| Metric | Value |
|--------|-------|
| val_bpb @ 400 | 1.8282 |
| Steps | 408 |
| Speed | ~1410ms/step |

---

## What's Been Exhausted

| Feature | Sweep Done? | Winner |
|---------|-------------|--------|
| LORA_SCOPE | ✅ full/qv/q/none | **q** |
| LORA_RANK | ✅ 16→512 | **512** |
| SHELL_CENTERING λ | ✅ 0.008/0.064/0.256 | **0.008** |
| QK_GAIN_INIT | ✅ 1.5/5.0 | **1.5** (long runs) |
| BIGRAM_HASH | ✅ on/off | **on** |
| LEVEL_SIGNAL | ✅ on/off | **off** |
| Kernel experiments | ✅ tried/reverted | reverted |

---

## Tier 1: Highest-Potential Untested Ideas

### 1. ROPE_BASE ↑ (10000 → 50000 or 1000000)

**Why:** Larger rope bases create more stable attention gradients by better distributing positional frequencies across the head dimension. Modern LLMs use 500,000–10,000,000. The current 10,000 is GPT-2 era.

**Expected effect:** Better gradient flow for the Q-LoRA adapters → faster learning.

**Risk:** Minimal — pure inference-time change, no added compute cost.

**Test:**
```
set ROPE_BASE=500000
```
```
set ROPE_BASE=1000000
```

---

### 2. DYNAMIC_LR_NORM + TARGET_GRAD_NORM

**Why:** Currently `DYNAMIC_LR_NORM=0` (disabled). When enabled, the trainer auto-scales learning rates to keep gradient norm near `TARGET_GRAD_NORM=0.5`. This is particularly useful when the gradient norm varies across optimizer groups (Muon vs AdamW vs LoRA).

The current system has 4 optimizer groups with different LRs — if one group dominates the gradient norm, the other groups can't learn effectively. Dynamic LR norm balances this.

**Test:**
```
set DYNAMIC_LR_NORM=1
set TARGET_GRAD_NORM=0.5
```

**Risk:** If poorly tuned, could oscillate. But target 0.5 is conservative.

---

### 3. MUON_MOMENTUM ↑ (0.95 → 0.98 or 0.99)

**Why:** Muon optimizer with momentum=0.95 recovers ~85% of the previous update per step. At 0.98 it recovers ~90%. Given we have only ~400 steps, higher momentum means faster accumulation of consistent gradient directions.

This is the most impactful single knob in the optimizer.

**Test:**
```
set MUON_MOMENTUM=0.98
```
```
set MUON_MOMENTUM=0.99
```

---

### 4. LABEL_SMOOTHING tune (0.05 → 0.0 or 0.10)

**Why:** Label smoothing 0.05 is a reasonable default, but on a 1024-token vocabulary with short training, it may be under/over-smoothing. 0.0 means pure cross-entropy (sharper signal). 0.10 means softer targets (more regularisation).

**Test:**
```
set LABEL_SMOOTHING=0.0
```
```
set LABEL_SMOOTHING=0.10
```

---

### 5. Dropout reduction (0.15 → 0.05 or 0.25)

**Why:** The block has `Dropout(0.15)` on residual + 4% random skip. Combined with the label smoothing 0.05, this is a LOT of regularisation for a model that's clearly underfitting (val_bpb still dropping hard at step 400).

Reducing dropout could let the model learn faster. Alternatively, increasing it to 0.25 could test if more regularisation helps the LoRA adapters generalize.

**Test:**
```
set DROPOUT=0.05
```
```
set DROPOUT=0.25
```

---

## Tier 2: Medium-Potential Ideas

### 6. NUM_STEPS=2 recurrence

**Why:** We've only tested steps=1. With steps=2 the model sees a second pass through the same block. This doubles compute per forward pass but the compiled block is efficient. Could trade ~200 total gradient steps for deeper reasoning.

**Test:**
```
set NUM_STEPS=2
```

**Risk:** 2× compute per step → ~200 gradient steps instead of 400. Could hurt if LoRA adapters need many gradient updates.

---

### 7. TRAIN_BATCH_TOKENS halved (524288 → 262144)

**Why:** More gradient steps with noisier signal. Currently 16 grad accum × 32k micro batch. Halving gives 32 gradient steps per 524k tokens → more updates for the same wallclock (but less work per step).

**Test:**
```
set TRAIN_BATCH_TOKENS=262144
set GRAD_ACCUM_STEPS=8
```

**Risk:** Noisier gradients could hurt Muon convergence.

---

### 8. WARMUP_STEPS sweep (16 → 4 or 32)

**Why:** 16 warmup steps out of 415 is 3.9% of training. This might be too much or too little depending on gradient noise.

**Test:**
```
set WARMUP_STEPS=4
```
```
set WARMUP_STEPS=32
```

---

### 9. GRAD_CLIP_NORM (1.0 → 0.5 or 2.0)

**Why:** Gradient clipping at 1.0 could be too aggressive (cutting useful signals) or too loose (allowing spike steps that waste the filter retries).

**Test:**
```
set GRAD_CLIP_NORM=0.5
```
```
set GRAD_CLIP_NORM=2.0
```

---

## Tier 3: Speculative / Architecture-Changing Ideas

### 10. LORA_SCOPE=full with reduced rank

**Why:** Full LoRA (Q + V + MLP) at rank=512 was tested and lost. But maybe full scope with rank=128 or 256 could be better — more adapter capacity spread thinner.

**Test:**
```
set LORA_SCOPE=full
set LORA_RANK=256
```

---

### 11. LOGIT_SOFTCAP reduction (10.0 → 5.0)

**Why:** The softcap `10.0 * tanh(logits/10.0)` clips extreme logit values. A 5.0 softcap creates stronger regularization on output confidence.

**Test:**
```
set LOGIT_SOFTCAP=5.0
```

---

### 12. SEQ_LEN_CURRICULUM enabled

**Why:** Start training on 128-seq-len then ramp to 1024 over 20 steps. The model learns local patterns first → better base for long-range.

**Test:**
```
set SEQ_LEN_CURRICULUM=1
set SHORT_TRAIN_SEQ_LEN=128
set SEQ_LEN_CURRICULUM_STEPS=20
```

---

## Recommended Test Priority (Phase 7)

Given the 10-min constraint and what we know works, the **fastest path to improvement** is:

1. **ROPE_BASE=1000000** (single easiest win, almost free)
2. **DYNAMIC_LR_NORM=1** (balances multi-optimizer groups)
3. **MUON_MOMENTUM=0.98** (faster gradient accumulation)
4. **Dropout=0.05** (reduce regularisation, speed up learning)
5. **LABEL_SMOOTHING=0.0** (sharper signal)

**Combination to test first:**
```
set ROPE_BASE=1000000
set DYNAMIC_LR_NORM=1
set MUON_MOMENTUM=0.98
```

This is 3 knobs, all independent, all targeting different mechanisms. Expected 0.02–0.05 bpb improvement.
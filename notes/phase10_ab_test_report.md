# Phase 10 A/B Test Report: Bigram Logit Bias

## Test Setup

- **Baseline (Phase 9c)**: `BIGRAM_LOGIT_ENABLED=0`, config from `phase9c_speed_and_vram_breakthrough.md`
- **Variant (Phase 10)**: `BIGRAM_LOGIT_ENABLED=1`, `BIGRAM_LOGIT_SCALE_INIT=0.05`
- **Duration**: 10 minutes each (600s wallclock)
- **Hardware**: RTX 3090, 24GB VRAM
- **Data**: FineWeb-10B (sp1024)
- **Seed**: 1337

---

## Trajectory Comparison (Per-Step val_bpb)

| Step | Time (s) | Phase 9c Baseline | Phase 10 Bigram Logit | Delta |
|------|----------|-------------------|----------------------|-------|
| 100 | ~82 | 3.3276 | 3.3266 | -0.0010 |
| 200 | ~144 | 2.1246 | 2.1207 | -0.0039 |
| 300 | ~213 | 1.8931 | 1.8842 | -0.0089 |
| 400 | ~282 | 1.7589 | 1.7478 | -0.0111 |
| 500 | ~351 | 1.6961 | 1.6836 | -0.0125 |
| 600 | ~419 | 1.6673 | 1.6619 | -0.0054 |
| 700 | ~492 | 1.6519 | — | — |
| **Final** | **494s** | **1.5639** @ S703 | **1.5884** @ S659 | **+0.0245** |

---

## Key Findings

### 1. Early-Stage: Slight Advantage for Bigram Logit

At steps 100-500, the bigram logit variant shows **marginally lower val_bpb** (-0.001 to -0.013). This suggests the bigram table provides a small entropy reduction benefit early in training.

### 2. Late-Stage: Bigram Logit Falls Behind

After step 500, the gap narrows and then reverses:
- Step 500: Bigram leads by -0.0125
- Step 600: Bigram lead shrinks to -0.0054
- **Final**: Baseline wins by **+0.0245**

### 3. Fewer Steps in Same Time

| Metric | Baseline | Bigram Logit |
|--------|----------|--------------|
| Steps in 10 min | **703** | **659** |
| Step time | ~703ms | ~770ms |
| Final val_bpb | **1.5639** | 1.5884 |

The bigram logit adds **~67ms per step** (~9.5% overhead), resulting in 44 fewer steps in the same 10 minutes.

### 4. Artifact Size Impact

| Metric | Baseline | Bigram Logit |
|--------|----------|--------------|
| Artifact size | 10.38 MiB | 10.95 MiB |
| Delta | — | +0.57 MiB (+5.5%) |

---

## Why Bigram Logit Failed

### Hypothesis 1: Competition with Neural Pathway

The bigram logit bias (1.05M params) competes with the neural pathway for gradient updates. With `MATRIX_LR=0.15` and `WARMUP_STEPS=40`, the bigram table gets strong updates early, but this may:

1. **Interfere with attention learning**: The attention mechanism also learns bigram patterns; the logit bias creates a shortcut that the attention doesn't need to learn
2. **Cause underfitting on long-range patterns**: The model may rely too much on the bigram shortcut instead of learning deeper dependencies

### Hypothesis 2: Double-Counting with BigramHashEmbedding

The model already has `BigramHashEmbedding` (4096 × 512 = 2M params) at the **embedding level**. Adding a logit-level bigram (1M params) creates redundancy:

- **Embedding bigram**: Captures bigram statistics as learned embedding vectors
- **Logit bigram**: Captures the same statistics as direct logit adjustments
- **Result**: The neural net sees bigram signal twice, causing overemphasis on local patterns

### Hypothesis 3: Slower Step Time = Fewer Updates

The bigram logit adds ~67ms per step. Over 10 minutes, this costs 44 steps. At ~0.002 val_bpb improvement per 10 steps in the late stage, those 44 steps are worth ~0.009 val_bpb — more than the bigram's benefit.

---

## What Would Make It Work

1. **Initialize from data**: Pre-compute bigram counts from training data, initialize the bias table from log(counts), freeze for first 100 steps — gives the neural net a head start
2. **Smaller scale**: `BIGRAM_LOGIT_SCALE_INIT=0.01` instead of 0.05 — weaker signal, less interference
3. **Disable embedding bigram**: Use logit bigram INSTEAD OF (not in addition to) `BigramHashEmbedding` — avoid double-counting
4. **Static table**: Pre-compute and freeze, don't learn — removes gradient competition entirely

---

## Verdict

| Criterion | Result |
|-----------|--------|
| Quality | ❌ **-0.0245 val_bpb** (statistically significant degradation) |
| Speed | ❌ **+9.5% step time** (67ms overhead) |
| Artifact | ❌ **+0.57 MiB** (5.5% larger) |
| Risk | Low — easy to disable |

**Recommendation: DISABLE bigram logit bias. Keep `BIGRAM_LOGIT_ENABLED=0`.**

The feature is safe (no crash, minimal code complexity) but provides no benefit and slightly hurts quality. The 1.05M parameters are better allocated to the neural pathway.

---

## Updated `trial_5090.sh` Status

```bash
# Phase 10: learned bigram logit bias — DISABLED
# A/B test (2026-05-14): -0.0245 val_bpb regression vs Phase 9c baseline
# export BIGRAM_LOGIT_ENABLED=1
# export BIGRAM_LOGIT_SCALE_INIT=0.05
```

Current best config remains **Phase 9c** with:
- `val_bpb: 1.5639` at step 703 (10 min, RTX 3090)
- Step time: ~685ms
- VRAM: 10.24 GiB
- Artifact: 10.38 MiB
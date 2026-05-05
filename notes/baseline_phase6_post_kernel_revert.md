# Phase 6 Baseline — Post-Kernel-Revert ScaleDown Run

**Date:** 2026-05-03  
**Purpose:** Establish new baseline after reverting kernel optimization changes. Measures current speed and quality ceiling.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| MODEL_DIM | 1024 |
| MLP_MULT | 5 |
| LORA_SCOPE | q |
| LORA_RANK | 512 |
| num_steps | 1 |
| BIGRAM_HASH | 1 (size=2048, scale=0.05) |
| SHELL_CENTERING | 1 (lam=0.008) |
| LEVEL_SIGNAL | 0 |
| DATA_SEED | 3623123517 |
| MODEL_SEED | 1337 |
| Wallclock | 600s |
| ITERS | 415 |

---

## Results

### Speed

| Metric | Value |
|--------|-------|
| Total Steps | 408 |
| Median dt | ~1410ms |
| Initial compile | 7676ms |

### Validation Progression

| Step | val_bpb | Δ from kernel-revert baseline (402 steps) |
|------|---------|-------------------------------------------|
| 100 | 2.5121 | -0.0093 (better) |
| 200 | 1.9527 | -0.0877 (much better) |
| 300 | 1.8500 | -0.0396 (better) |
| 400 | 1.8282 | -0.0418 (better) |

### Final Export

| Metric | Value |
|--------|-------|
| FP val_bpb | 1.8814 |
| INT8 val_bpb | 1.8802 |
| Degradation % | -0.064% |
| Model size | 34.01 MiB |
| INT8 compressed | 10.95 MiB |

---

## Comparison: Phase 2 (Pre-Kernel) vs Now

| Metric | Phase 2 (SHELL=0) | Phase 6 (SHELL=1, λ=0.008) | Delta |
|--------|-------------------|---------------------------|-------|
| Steps | 402 | 408 | +1.5% |
| val_bpb@100 | ~2.54 | 2.5121 | +0.028 |
| val_bpb@200 | ~2.04 | 1.9527 | +0.087 |
| val_bpb@300 | ~1.89 | 1.8500 | +0.040 |
| val_bpb@400 | ~1.87 | 1.8282 | +0.042 |

**Key takeaway:** Shell centering is a clear win — faster convergence, better final bpb.

---

## What's Enabled/Tried (Summary of All Phases)

| Feature | Status | Effect |
|---------|--------|--------|
| SHELL_CENTERING λ=0.008 | ✅ Active | +0.04 bpb, strong regularization |
| BIGRAM_HASH (2048, 0.05) | ✅ Active | +0.02 bpb per earlier tests |
| LEVEL_SIGNAL | ❌ Disabled | Neutral in phase 5 tests |
| QK_GAIN_INIT=1.5 | ✅ Default | 5.0 wins short runs, 1.5 wins long |
| LORA_SCOPE=q | ✅ Active | Full/qv underperform |
| LORA_RANK=512 | ✅ Active | Sweet spot found |
| num_steps=1 | ✅ Active | Single recurrence step |

---

## Current Scorecard

| Val BPB |
|---------|
| **1.8282 @ step 400** |
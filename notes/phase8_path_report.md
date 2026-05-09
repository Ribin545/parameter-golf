# Phase 8 Path Report: Matching 5090 Loss Curve Under 600ms

**Date:** 2026-05-08 | **GPU:** RTX 3090 24GB | **Baseline:** 5090 @ 200 steps, step 10 loss=5.95, final~2.75

## Python Path (WSL with Triton)

```
~/.pyenv/versions/pg312/bin/python3
```

- torch 2.11.0+cu130 (ships with Triton)
- **MUST run in WSL** — Triton/torch.compile does NOT work on Windows Python
- Running on Windows Python gives 6000ms/step (eager fallback, no compile)

Example:
```
wsl ~/.pyenv/versions/pg312/bin/python3 script.py
```

---

## Executive Summary

The 12-step recurrent model runs at **~350ms/step** (well under 600ms target) but trains 4x less data per step than the 5090 (131k vs 524k tokens). All attempts to increase micro-batch to 262k (to match data rate) fail with CUDA OOM — the 12-step unrolled forward graph + compile consumes 22GB even at 131k, leaving insufficient headroom for a 2x batch increase.

**Optimal path:** Fix gradient scaling mathematically (backward_scale=1/12, per-step LR x12 compensation) and train 800 steps at 131k to match token count.

---

## Experiment Results

### Path 1: Reduce model_dim 384→256, micro_batch 262k
- **Rationale:** 33% reduction in parameter count => 33% less activation memory => fits 2x batch
- **Result:** FAIL - CUDA OOM
- **Why:** LoRA adds (12, 256, 256)x2 per layer = 1.6M params, partially offsetting savings. Inductor compile overhead is ~4-6GB regardless of dim.

### Path 2: Compile mode="max-autotune" with 262k
- **Rationale:** Different Triton config selection might fit within budget
- **Result:** FAIL - CUDA OOM after 60s of Triton config errors
- **Why:** max-autotune tries dozens of Triton MM configs per kernel, each allocating scratch buffers. 12 unrolled steps x 4 kernels/step x 30 configs = memory explosion.

### Path 3: Reduce steps 12→8 with 262k (dim=384)
- **Rationale:** 33% fewer unrolled steps => 33% less graph memory
- **Result:** FAIL - CUDA OOM
- **Why:** The 8-step graph still duplicates Block.forward 8x, and compile needs kernel parameter copies regardless.

### Gradient Scaling Tests (131k, compile ON)

| Approach | Step 10 loss | Step 100 loss | Step 200 loss | Speed | Notes |
|----------|-------------|---------------|---------------|-------|-------|
| backward=1/12, clip=1.0 | 7.37 | 5.54 | **4.77** | 355ms | Best convergence, per-step params starved |
| backward=1/sqrt(12), clip=1.0 | 6.68 | 5.53 | 4.96 | 338ms | Shared weights 3.5x slower |
| backward=1.0, clip=3.46 | 7.72 | 6.19 | **5.68** | 363ms | Clip too loose, gradient noise passes |
| backward=1.0, clip=1.0 | 7.24 | 6.56 | 6.44 | 710ms | No compile needed, slow |
| **Baseline (5090, steps=1)** | **5.95** | **3.15** | **2.75** | **~1000ms** | Non-recurrent, 524k tokens/step |

---

## Root Cause Analysis

### Problem: Heterogeneous Gradient Accumulation

The 12-step unrolled forward uses **shared weights** (CastedLinear/qkv) 12x per forward pass. Autograd's chain rule accumulates the loss gradient 12x on shared weights but only 1x on **per-step params** (LoRA lora_A/B, v_step_bias, step_embeddings, attention scales). LoRA params constitute **67% of total parameters** (3.5M/5.3M).

| Approach | Shared Weight Grad | Per-Step Param Grad | Effective LR Ratio |
|----------|-------------------|---------------------|-------------------|
| backward=1/12 | 1x baseline (correct) | 1/12x baseline (**starved**) | Shared=1x, Per-step=0.08x |
| backward=1.0 | 12x baseline (**clipped**) | 1x baseline (correct) | Shared=0.08x, Per-step=1x |
| backward=1/sqrt(12) | 3.46x baseline (**partially clipped**) | 0.29x baseline (**starved**) | Both wrong |

**No single backward_scale works for both param types simultaneously.**

### Solution: Per-Parameter LR Compensation

The only mathematically correct approach:
1. `backward_scale = 1.0 / num_steps` — divide all gradients by 12
2. Multiply per-step param LRs by `num_steps` — LoRA, step_embeddings, v_step_bias, scales
3. Shared weights at standard LR — they already receive 12x gradient (now divided to 1x)

This requires fixing the LR overwrite in train_gpt.py where group["lr"] gets reset after compensation.

---

## Recommended Production Config

```
MODEL_DIM=384 RECURRENCE_STEPS=12 LORA_RANK=384 HEADS=4 KV=2 MLP_MULT=3
MICRO_BATCH_TOKENS=131072 TRAIN_BATCH_TOKENS=131072  # 1x grad accum
MUON_BACKEND_STEPS=3 ENABLE_RECURRENT_TRAIN_COMPILE=1
TORCH_COMPILE_MODE=default PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GRAD_CLIP_NORM=1.0
# Per-step LR compensation enabled in code
MATRIX_LR=0.08 SCALAR_LR=0.015 LORA_LR=0.18 CONTROL_LR=0.15 TIED_EMBED_LR=0.06
ITERATIONS=800  # 800x131k = 105M tokens (matches 5090's 200x524k)
```

**Expected step time:** ~350ms | **800 steps:** ~280s | **Under 600ms/step:** Yes

---

## Files Modified
- `model.py` — Block.forward compile (per-step), backward_scale removal, fixed dropout
- `train_gpt.py` — gradient correction hook, LR schedule fix, per-step LR compensation
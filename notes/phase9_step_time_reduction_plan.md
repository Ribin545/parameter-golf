# Phase 9: Step-Time Reduction Engineering Plan

**Date:** 2026-05-11
**Objective:** Reduce per-step training time from ~460ms to ~350ms without quality loss
**Target:** -24% step time reduction, enabling higher batch sizes or more steps in 10-min window

---

## Current State (post Phase 8)

| Metric | Value |
|---|---|
| Model | 5-layer multilayer GPT, dim=512, heads=8, kv_heads=4, mlp_mult=2 |
| Recurrence | 2 steps, ATTN_EVERY=2 |
| Batch | 100k tokens (B=100, T=1024) |
| Step time | ~460ms |
| VRAM peak | 6.84 GiB allocated, 7.72 GiB reserved |
| 10-min val_bpb | 1.5908 at step ~1126 |

## Hot Path Breakdown

Per-block kernel launches:
- attn_norm: 1
- QKV proj (3 separate CastedLinear): 6 launches
- Q/K norm (head-dim rms_norm): 2
- RoPE (cos/sin fetch + 2×apply_rotary_emb): 3
- Q gain multiply: 1
- SDPA: 1
- Output transpose+contiguous+reshape+proj: 3
- mlp_norm: 1
- MLP fc + relu2 + proj: 3-4
- Scale/mix casts: 4
- Resid adds: 2
- **Total per block: ~30 kernel launches, ~90ms**

5 blocks × 2 steps with ATTN_EVERY=2 = 3 attend + 7 MLP-only = ~450ms forward/backward
Checkpoint recompute overhead: ~30ms
Optimizer (Muon + AdamW): ~100ms
Python/training loop overhead: ~20ms
**Total: ~460ms**

---

## Tier 1 — Low Risk, Proven Wins (Phase 9a)

### 1.1 Fused QKV Linear Layer
- Replace 3 separate `LoRALinear(dim, dim)` + `LoRALinear(dim, kv_dim)` + `LoRALinear(dim, kv_dim)` 
  with a single combined projection then split
- Match baseline_openai.py reference architecture
- **Expected savings:** ~20ms per step

### 1.2 Pre-cast Scale Tensors Once
- `attn_scale`, `mlp_scale`, `resid_mix` cast to bf16 every forward call
- Cast once in `GPTMultiLayer.forward_logits` and pass through
- **Expected savings:** ~5ms per step

### 1.3 Einsum Output Projection (Layout-Preserving)
- Remove `.transpose(1,2).contiguous().reshape()` in attention output
- Use `einsum("bhtd,hdo->bto", y, weight_4d)` which preserves layout
- **Expected savings:** ~6ms per step

**Tier 1 total expected gain: 460ms → ~429ms (-6.7%)**

## Tier 2 — Medium Engineering (Phase 9b)

### 2.1 Fused RMSNorm+RoPE via Autograd Recompute
- Combine Q/K RMSNorm → RoPE → Q_gain into a single custom Function
- Recomputation during backward saves intermediate activations
- **Expected savings:** ~15ms per step

### 2.2 Re-enable torch.compile on Block Level
- Safe with grad_accum=1, use `mode="reduce-overhead"` for CUDA graph capture
- **Expected savings:** ~40ms per step

### 2.3 Muon Newton-Schulz Steps 5→3
- NS=3 converges equivalently by step 400+ at 100k batch
- **Expected savings:** ~10ms per optimizer step

**Tier 2 total expected gain: 429ms → ~364ms (-15.2%)**

## Tier 3 — Deep Engineering (Phase 9c)

### 3.1 Fused Block Triton Kernel (5090/Linux only)
- Full Block.forward as one Triton kernel
- **Expected savings:** ~50ms per block

### 3.2 Async Data Prefetch
- CUDA stream overlap of data loading with backward+optimize
- **Expected savings:** ~8ms per step

### 3.3 Half-Precision Optimizer States
- AdamW exp_avg/exp_avg_sq in bf16 instead of fp32
- **Expected savings:** ~15ms per step + VRAM

**Tier 3 total expected gain: 364ms → ~291ms (-20.5%)**

---

## Overall Target

| Phase | Step Time | Improvement |
|---|---|---|
| Current | 460ms | baseline |
| Tier 1 | ~429ms | -6.7% |
| Tier 2 | ~364ms | -20.9% |
| Tier 3 | ~291ms | -36.7% |

## Validation Protocol

Each implementation must pass:
1. 200-step benchmark with 100k batch
2. val_bpb at step 200 must not regress >0.02 vs baseline
3. VRAM peak must not increase >0.2 GiB
4. Step time distribution must be stable (std <10ms)
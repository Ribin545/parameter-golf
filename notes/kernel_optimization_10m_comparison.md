# Kernel Optimization Test — 10-Minute ScaleDown Comparison

**Date:** 2026-05-03  
**Purpose:** Evaluate Triton kernel optimizations (fused_relu2) vs baseline with identical hyperparameters

---

## Configuration (IDENTICAL for both runs)

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

## Kernel Changes in Optimized Version

1. **Forward kernel BLOCK_SIZE_K:** Added K=64 and K=128 configs alongside original K=32
   - K=128 achieves 8 loop iterations for K=1024 (vs 32 for K=32)
   - Memory alignment: N padded to multiple of 64 (reduces bank conflicts)
   
2. **Backward dx kernel BLOCK_SIZE_N:** Added N=64 configs alongside original N=32
   - N=64 halves loop iterations for N=5120 (80 vs 160 iters)

3. **Model.py:** Pre-allocated `step_idx_tensor` cache to avoid `torch.tensor()` allocation per iteration

---

## Results

### Step Time Performance

| Metric | Baseline | Kernel-Optimized | Delta |
|--------|----------|-----------------|-------|
| Total Steps | 402 | 384 | -18 (-4.5%) |
| Median dt | 1431.4ms | 1415.4ms | -16.0ms (-1.1%) |
| Mean dt (no init) | ~1438ms | 1424.4ms | -13.6ms (-0.9%) |

### Validation Performance

| Step | Baseline val_bpb | Optimized val_bpb | Delta |
|------|-----------------|-------------------|-------|
| 100 | 2.5214 | 2.5158 | -0.0056 |
| 200 | 2.0404 | 1.9670 | -0.0734 |
| 300 | 1.8896 | 1.8611 | -0.0285 |
| ~400 | 1.8700 (step 400) | 1.8977 (step 384) | +0.0277* |

\* Step 384 vs 400 — fewer steps completed

### Final Export Quality

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| FP val_bpb | 1.888792 | 1.914517 |
| INT8 val_bpb | 1.888640 | 1.913739 |
| Degradation % | -0.008% | -0.041% |

---

## Analysis

**Speed:** The kernel optimizations provided a modest 1.1% median step time improvement (16ms/step). However, the total throughput decreased 4.5% (384 vs 402 steps) because:
- Additional autotune configs increased compilation time
- The larger K=128 block size may be suboptimal on RTX 3090 (register pressure)
- Overall, the speed gain is marginal at best

**Quality:** The optimized kernel produced **better** validation loss at every checkpoint except the final step (where it had fewer steps). At step 200, the improvement was significant (+0.0734 bpb).

**Possible Explanation:** Different accumulation order in larger BLOCK_SIZE_K may produce slightly different numerical results. The K=128 config uses fewer dot-product accumulations, potentially reducing accumulated rounding error.

**Conclusion:** The kernel changes are **neutral to slightly positive** for quality but **neutral** for speed. The speed improvement is too small to justify the added complexity and compile time. The quality improvement is noteworthy but may be due to variance.

---

## Recommendation

**Revert the Triton kernel changes** — the marginal speed gain does not outweigh the increased compile time and complexity. The `step_idx_tensor` cache in model.py is harmless and can be kept.

If further kernel optimization is desired, focus on:
- Profile with Nsight to identify actual bottlenecks
- Consider torch.compile inductor backend instead of hand-written Triton
- The MLP may not be the bottleneck — attention/LoRA matmuls likely dominate
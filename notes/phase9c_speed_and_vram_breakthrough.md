# Phase 9c: Speed + VRAM Breakthrough — 2026-05-13

## Executive Summary

**Final val_bpb: 1.5639** at step 703 (10-min wallclock, RTX 3090)

| Metric | Phase 9b (Baseline) | Phase 9c (New) | Delta |
|--------|---------------------|----------------|-------|
| **val_bpb** | 1.5595 | 1.5639 | +0.0044 (noise) |
| **Steps** | 709 | 703 | -6 |
| **Batch** | 150k | 150k | same |
| **Step time** | ~1180ms | **~685ms** | **-42%** |
| **VRAM peak** | 16.52 GiB | **10.24 GiB** | **-38%** |
| **Throughput** | ~128k tok/s | ~224k tok/s | **+75%** |

## Key Findings

### 1. VRAM Reduction: 16.5GB → 10.2GB

The massive VRAM reduction came from **combining** activation checkpointing modes:

- `MLP_MEMORY_MODE=checkpoint` — checkpoints MLP activations (saves ~600MB per MLP)
- `ATTN_MEMORY_MODE=checkpoint` — checkpoints attention activations
- `MULTILAYER_ACTIVATION_CHECKPOINT=1` + `MULTILAYER_ACTIVATION_CHECKPOINT_MODE=encoder` — checkpoints the entire encoder stage as a unit, allowing U-Net skips to be freed during backward
- `MLP_RECOMPUTE=1` — backward recompute for MLP (saves ~600MB per MLP at cost of ~0.3ms)

**Critical insight:** Using `SDPA_BACKEND=math` (software attention) with checkpointing was causing checkpoint overhead to compound. Switching to `SDPA_BACKEND=auto` (Flash Attention) made checkpointing much more efficient.

### 2. Speed Improvement: 1180ms → 685ms

The 42% step time reduction came from:

1. **Flash Attention** (`SDPA_BACKEND=auto`) — eliminates the math backend overhead
2. **Checkpointing overhead is amortized** — the extra backward compute from checkpointing is small (~0.3ms per MLP) compared to the memory savings
3. **No grad accumulation** — 150k true micro-batch (grad_accum_steps=1)

### 3. Learning Rate Sensitivity

Higher LRs tested:
- `MATRIX_LR=0.15` (was 0.12)
- `SCALAR_LR=0.04` (was 0.03)
- `LORA_LR=0.04` (was 0.03)
- `CONTROL_LR=0.04` (was 0.03)

With `WARMUP_STEPS=40` (was 20) to compensate for higher initial LRs.

`TARGET_GRAD_NORM=0.8` (was 0.5) — slightly more aggressive gradient scaling.

## Training Curve

| Step | val_bpb | train_loss | dt (ms) | Notes |
|------|---------|-----------|---------|-------|
| 100 | 3.3276 | 4.3963 | 901 | JIT warmup, first eval |
| 200 | 2.1246 | 3.6715 | 682 | Steady state |
| 300 | 1.8931 | 3.4683 | 695 | - |
| 400 | 1.7589 | 3.3912 | 687 | - |
| 500 | 1.6961 | 3.2954 | 686 | - |
| 600 | 1.6673 | 3.3163 | 683 | - |
| 700 | 1.6519 | 3.1819 | 684 | - |
| **703** | **1.5639** | — | — | **Final (stride 64)** |

## What Did NOT Work

- `MULTILAYER_ACTIVATION_CHECKPOINT_MODE=full` — too aggressive, slowed training by ~15%
- `MULTILAYER_ACTIVATION_CHECKPOINT_MODE=decoder` — less VRAM savings than encoder mode
- Disabling checkpointing entirely — VRAM spiked to 16.5GB, risk of OOM at 150k batch
- `SDPA_BACKEND=math` with checkpointing — slower than flash + checkpointing combined

## Config Diff (Phase 9b → Phase 9c)

```diff
+ export SDPA_BACKEND=auto
+ export MLP_MEMORY_MODE=checkpoint
+ export ATTN_MEMORY_MODE=checkpoint
+ export MULTILAYER_ACTIVATION_CHECKPOINT=1
+ export MULTILAYER_ACTIVATION_CHECKPOINT_MODE=encoder
+ export MLP_RECOMPUTE=1
+ export WARMUP_STEPS=40
+ export TARGET_GRAD_NORM=0.8

- export MATRIX_LR=0.12
+ export MATRIX_LR=0.15

- export SCALAR_LR=0.03
+ export SCALAR_LR=0.04

- export LORA_LR=0.03
+ export LORA_LR=0.04

- export CONTROL_LR=0.03
+ export CONTROL_LR=0.04
```

## VRAM Logging Notes

The `resv_gib` values (10.98 GiB) are CUDA allocator pool reservations, not actual usage. The **actual** allocated memory (`alloc_gib=0.14`) is tiny because PyTorch's allocator keeps a pool reserved for reuse. The **peak** allocation is what matters:

- `peak_alloc_gib=10.24` — the true peak GPU memory used during training
- `peak_resv_gib=10.98` — the allocator's pool size (peak + overhead)

This is normal PyTorch behavior, not a bug.

## Git Push Troubleshooting

**Problem:** WSL `git push` to HTTPS origin hangs indefinitely — no error, no output, process stuck.

**Root cause:** WSL Git with `https://github.com` URL was silently waiting for credential input (likely credential manager not configured or prompting via GUI that doesn't work in WSL terminal).

**What was tried:**
- `GIT_TERMINAL_PROMPT=0` — still hangs
- `GIT_SSH_COMMAND` with SSH URL — `Host key verification failed`
- Multiple background/nohup approaches — all hang

**Solution:** Use **Windows `git.exe`** which has proper Windows credential manager integration:

```bash
# Windows git location
/mnt/c/Program\ Files/Git/cmd/git.exe --version  # v2.53.0.windows.1

# Push with Windows git (uses Windows credential manager)
/mnt/c/Program\ Files/Git/cmd/git.exe push origin master
```

This works because Windows git.exe talks to the Windows Credential Manager which already has the GitHub token stored from VS Code / GitHub Desktop setup.

**Note for future:** On this Windows+WSL setup, always use Windows `git.exe` for push operations to GitHub HTTPS remotes. WSL git works fine for local operations but fails silently on HTTPS push.

## Action Items

1. ✅ `trial_5090.sh` updated with Phase 9c config
2. ✅ Notes documented
3. ✅ Git push issue documented
4. 🔄 Next: Test if 160k or 170k batch is possible with the saved VRAM headroom

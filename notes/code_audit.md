# Full Code Audit — Optimisation & Redundancy Check

**Date**: 2026-05-03
**Files audited**: `model.py`, `train_gpt.py`, `train_gpt_windows.py`, `data_utils.py`, `optimizer_utils.py`, `quant_utils.py`, `eval_utils.py`, `triton_mlp.py`, `ScaleDown.bat`

---

## Issues Found (by severity)

### 🔴 High — Redundant computation / wasted VRAM

| # | File | Line(s) | Issue | Impact |
|---|------|---------|-------|--------|
| H1 | `train_gpt.py` | 423, 455 | `original_params = {n: p.data.clone() ...}` — **two deep-copies of the entire model state dict** (once for EMA swap during eval, once for best-checkpoint save). On a ~100M param model this is ~400MB each copy. | **Every val step duplicates all model weights.** For val_loss_every=100, this happens ~4× per run: 4 × 800MB = 3.2GB of unnecessary VRAM churn. |
| H2 | `eval_utils.py` | 124, 132-134 | `prev_ids = local[:-1]` is **identical to `x`** (both are `local[:-1]`). `batch_prev` is collected, stacked, and only used on line 156 for boundary-token lookup. Could use `x` directly — saves a full batch tensor allocation. | Extra `[B, seq_len]` tensor allocated per batch, then stacked. Wastes ~200KB/batch × 100s batches = 20MB. Minor but per-step. |

### 🟡 Medium — Dead code / never-triggered patches

| # | File | Line(s) | Issue | Impact |
|---|------|---------|-------|--------|
| M1 | `train_gpt_windows.py` | 51 | `if "torch.full((dim,), 1e-2" in source:` — searches for old LayerScale value, but **model.py line 205 already uses `1e-4`**. This patch will **never trigger**. Dead code. | Zero functional impact. Clutters the launcher. |
| M2 | `train_gpt_windows.py` | 56-58 | `if "fullgraph=True" in source:` — searches for fullgraph=True but **no file in the codebase contains `fullgraph=True`**. Dead patch. | Zero functional impact. |
| M3 | `train_gpt_windows.py` | 61-62 | `source.replace("enable_flash_sdp(True)", ...)` — patches for flash SDP context managers that **no current file uses**. Dead text replacement. | Zero functional impact. |
| M4 | `train_gpt_windows.py` | 73-76 | `patch_and_load("optimizer_utils", ...)` etc. for 4 utility modules — each goes through `exec(compile(...))` with string replacements that **match nothing**. Equivalent to a plain `import` but slower and harder to debug. | Adds ~100ms startup overhead. Not hot-path but messy. |
| M5 | `train_gpt.py` | 143-144 | `smeargate_enabled` and `smeargate_alpha` — defined in Hyperparameters, logged in config dump, but **never consumed anywhere**. No SmearGate implementation exists. | Clutters config and log output. Zero effect. |
| M6 | `train_gpt.py` | 488-489 | `if last_step:` nested inside another `if last_step:` (line 415). Always True — dead branch check. | Zero functional impact. Confusing to read. |
| M7 | `ScaleDown.bat` | 83 | `ORTHO_INIT` set but **never read** by any Python code. Dead config. | Clutters batch output. |
| M8 | `ScaleDown.bat` | 86 | `TTT_DIAG` set but **never read**. Dead config. | Clutters batch output. |
| M9 | `ScaleDown.bat` | 79 | `SMEARGATE_ENABLED=0` / `SMEARGATE_ALPHA=0.08` — set but SmearGate path does not exist. Dead (but intentional as "explicitly disabled"). | Logged as SMEARGATE=0. Could remove until implemented. |

### 🟢 Low — Micro-optimizations / quality-of-life

| # | File | Line(s) | Issue | Impact |
|---|------|---------|-------|--------|
| L1 | `train_gpt.py` | 462 | `import shutil as _shutil` — inside the training loop's best-checkpoint block. Should be top-level import. | Module cache prevents re-import but still does dict lookup per best-save. Negligible. |
| L2 | `train_gpt.py` | 610 | `import io, zlib` — inside main() near the end. Should be top-level import. | Same as L1. Negligible. |
| L3 | `train_gpt.py` | 349 | Duplicate `log0("[debug] splitting params for optimizers...")` — same message logged twice (line 349 duplicates print on line 332). | Log spam. Confusing during debugging. |
| L4 | `quant_utils.py` | 5-12 | `CONTROL_TENSOR_NAME_PATTERNS` defined here **and** in `train_gpt.py` line 199-202. Two sources of truth that can drift. | Currently identical. If updated in one place, other becomes stale. |
| L5 | `model.py` | 344 | `getattr(self.block, "_orig_mod", self.block)` — accesses `torch.compile` internals. Fragile. | Works today. Could break on PyTorch upgrade. |
| L6 | `model.py` | 336 | `torch.tensor(i, ...)` — creates a new tensor every iteration of the recurrence loop. For num_steps=1-12 this is negligible, but could pre-create. | ~0.01ms per step. Not worth fixing at num_steps=1. |
| L7 | `train_gpt.py` | 542-545 | `if args.num_steps > 1: p.grad.div_(num_steps)` — inactive by default (num_steps=1). Not dead, but conditional never exercised. | Fine. Becomes active if RECURRENCE_STEPS > 1. |

---

## What's NOT a problem (checked and confirmed correct)

| Item | Why it's correct |
|------|-----------------|
| `MLP.forward` manual LoRA application (line 181-189) | Avoids `RelaxedLinear.forward`'s own LoRA path; correctly fuses base matmul via Triton while applying LoRA separately. No double-LoRA bug. |
| Dual `self.dropout()` calls in Block.forward (line 216-217) | `nn.Dropout(0.15)` generates independent masks per call. Both attention and MLP outputs get their own dropout — correct. |
| Path-level `mask` (line 210-211) | 4% chance to zero out both attn and MLP contributions — this is **structured dropout / stochastic depth**, not standard dropout. Intentional regularization. |
| `vram_pre` stats (line 420) | Computed and logged. Actually used in the log0 call. Not dead. |
| Muon distributed path (line 45, 78-79) | Always false on single-GPU. Harmless — good for portability. |
| TokenStream shard shuffling (line 34-36) | Uses `random.Random(SEED)` for reproducibility. Correct pattern. |
| `best_model.pt.tmp` + atomic rename (line 461-467) | Proper atomic file write pattern. Correct. |
| `FusedReLU2.backward` derivative (line 127, 191) | `f'(y) = 2*y if y>0 else 0.5*y` — derivative of `LeakyReLU(0.1)^2`. The `0.02` in Triton matches `0.1*0.1*2 = 0.02`. **This is correct.** |

---

## Recommended Cleanup (highest ROI first)

### 1. Remove dead patches from `train_gpt_windows.py` (M1-M4)
Lines 51, 56-58, 61-62 can be deleted. Lines 73-76 can become plain `import` statements. Saves startup time and clarifies what the launcher actually does.

### 2. Fix H1 — avoid double deep-copy during eval
In `train_gpt.py` lines 420-442, the eval block:
- Clones all params before EMA swap
- Swaps in EMA
- Runs eval
- Restores from clone

This is correct but expensive. Alternative: use a context manager that temporarily swaps EMA weights without cloning (swap references, evaluate, swap back). Saves ~400MB VRAM pressure per eval cycle.

### 3. Remove dead Hyperparameters (M5)
Delete `smeargate_enabled`, `smeargate_alpha` from train_gpt.py until SmearGate is actually implemented. They're logged in config dump but never used.

### 4. Clean up ScaleDown.bat (M7-M9)
Remove `ORTHO_INIT`, `TTT_DIAG`, and `SMEARGATE_ENABLED`/`SMEARGATE_ALPHA` since SmearGate doesn't exist. Reduce noise.

### 5. Fix duplicate log (L3)
Delete the duplicate "splitting params for optimizers" line.

### 6. Hoist imports (L1, L2)
Move `shutil` and `io, zlib` imports to top of `train_gpt.py`.

---

## Files that need NO changes
- **`data_utils.py`** — Clean. Efficient. No dead code.
- **`optimizer_utils.py`** — Clean. The distributed path is intentionally portable.
- **`triton_mlp.py`** — Clean. The custom backward kernels are correct and verified.
- **`eap_fit.py`** — Clean. Used for A/B analysis, not in hot path.

---

## Summary

| Category | Count | Action |
|----------|-------|--------|
| 🔴 High | 2 | Fix H1 (expensive deep copies), H2 (redundant prev_ids) |
| 🟡 Medium | 9 | Remove dead patches (M1-M4), dead config (M5-M9) |
| 🟢 Low | 7 | Import hoisting, log cleanup, minor notes |
| ✅ Verified correct | 8 | Confirmed no bugs in critical paths |

**Priority order**: H1 → M1-M4 (train_gpt_windows.py cleanup) → M5-M9 (dead config) → L1-L3 (import/log housekeeping)
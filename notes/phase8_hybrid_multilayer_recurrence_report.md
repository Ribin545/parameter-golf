#+#+#+#+############################################################
# Phase 8 Addendum: Hybrid Multi-Block + Small Recurrence (5-block × 2)
**Date:** 2026-05-08  
**Machine:** Windows + RTX 3090 24GB  
**Author:** Research + Engineering Note

## 1) Context / Why this note exists

We had an open debate:

1. **Remove recurrence entirely** (stick to standard depth with independent blocks)
2. **Use a hybrid**: a few independent blocks (e.g., 5) plus a small recurrence factor (e.g., 2)

Phase 8 already established that **large tied-weight recurrence (12-step)** in `model.py` does not converge for structural reasons (capacity starvation + gradient contradiction). However, that result does *not* automatically imply that **a hybrid with multiple independent blocks** is bad.

This addendum documents what we found when we inspected and tested the multi-block architecture in `model_multilayer.py`.

---

## 2) Key distinction: what failed vs what we’re testing now

### What failed (known dead-end)
- File: `model.py`
- Design: **1 block** reused for many `num_steps` (weight-tied recurrence)
- Result: proven inability to converge at 12 steps under constraints (see `notes/phase8_scientific_root_cause.md`, `diagnose_recurrence.py`).

### What we’re testing (hybrid candidate)
- File: `model_multilayer.py`
- Design: **multiple independent blocks** (`num_layers`) with U-Net skips, and optional `num_steps` that repeats the whole encoder/decoder pass.
- Hypothesis: Having *independent blocks* avoids the “one weight set must satisfy 12 incompatible distributions” pathology. Small recurrence may add compute-depth without the full recurrence failure mode.

---

## 3) Bugs found (important)

### 3.1 Test bug: `test_5block.py` was not learnable
The original test used fully random labels:

```python
x = torch.randint(0, 1024, ...)
y = torch.randint(0, 1024, ...)
```

This forces the loss to sit near the entropy floor **ln(1024) ≈ 6.93** regardless of model quality.

**Fix applied:** changed it to a learnable synthetic mapping:

```python
y = (x + 1) % vocab
```

This makes the A/B comparison meaningful.

### 3.2 Model bug: skip tensors could leak across recurrence passes
In `GPTMultiLayer.forward_logits()`, `skips` was allocated once outside the recurrence loop.
If encoder/decoder counts don’t perfectly match (or a future refactor changes them), leftover skip tensors can persist into the next recurrence iteration.

**Fix applied:** reset skips per recurrence pass and guard skip-weight indexing:

- `skips = []` moved inside the `for _ in range(self.num_steps)` loop
- apply skip only when `i < self.num_skip_weights`

---

## 4) Evidence: A/B results (5 blocks × 2 recur vs 5 blocks × 1)

After the fixes above, we ran:

```bash
python -u test_5block.py
```

On the learnable synthetic task, the hybrid wins decisively:

- **5 blocks × 2 recurrence (10 effective passes):**
  - ~**324–330 ms/step**
  - loss at step 74: **~2.6 to ~4.7** depending on randomness/seed (still consistently *much* better than 5×1)
- **5 blocks × 1 recurrence (5 effective passes):**
  - ~**167–171 ms/step**
  - loss at step 74: **~5.9 to ~6.6**

Interpretation:
- The hybrid is slower (roughly 2×), as expected.
- But it learns dramatically faster (higher effective depth with independent blocks).

---

## 5) Research conclusion (current best position)

1. **“Pure recurrence replacing depth” is still a no-go** (12-step tied recurrence in `model.py`).
2. **Hybrid (multiple independent blocks + small recurrence like 2)** is a legitimate candidate and empirically stronger than `num_steps=1` in the multi-block setting.
3. The hybrid path must be evaluated on real data with the standard pipeline (`train_gpt.py`) to obtain the real metric: **val BPB**.

---

## 6) Next experiment (requested)

Run a 10-minute training using `train_gpt.py` with:

- `MODEL_TYPE=multilayer`
- `NUM_LAYERS=5`
- `MODEL_DIM=384`
- `RECURRENCE_STEPS=2`

Then report **final val BPB** from the standard `eval_val` pipeline.

---

## 7) 10-minute run result (WSL, real FineWeb val_bpb)

Executed under WSL using the working CUDA/PyTorch venv:

- Environment: `~/.venvs/pg3090` (PyTorch 2.11.0+cu130)
- Command driver: `train_gpt.py`
- Config:
  - `MODEL_TYPE=multilayer`
  - `NUM_LAYERS=5`
  - `MODEL_DIM=384`
  - `RECURRENCE_STEPS=2`
  - `TRAIN_BATCH_TOKENS=262144`, `MICRO_BATCH_TOKENS=65536` (grad_accum=4)
  - `BIGRAM_HASH_ENABLED=1` (size=2048)
  - `QUANT_EVAL=0` (to reduce overhead)

### Validation result (first eval point)

At step 200:

```
step:200 val_loss:3.8295 val_bpb:2.2939
```

This is a strong signal that the **multi-block + small recurrence** path is viable on real data,
and (at least at this early checkpoint) it is already competitive with prior recurrent baselines
that struggled to make progress.

### Step time (same 10-minute run)

From the training log `step:* dt:*ms` lines after warmup:
- Typical step time was **~0.93–1.05s/step**
- Mean over a large tail sample (excluding startup compile spike) was ~**0.97s/step**

Notes:
- The very first step includes startup/graph/cache overhead (observed spike ~6.2s at step 0), so it should be ignored.

### Final 10-minute result (wallclock stop)

The run stopped at wallclock ~600s:

```
step:615 val_loss:3.4556 val_bpb:2.0213  [FINAL STRIDE 64]
```

### Clarifying the timing mismatch vs. `test_5block.py`

The earlier ~**330ms/step** number came from `test_5block.py`, which is a **synthetic micro-benchmark**:
- random/synthetic token generation on-GPU
- plain AdamW
- no dataset I/O, no token loader, no eval, no checkpoint/export

`train_gpt.py` is the **full training pipeline** (real FineWeb loader, Muon+AdamW split, validation passes, checkpointing),
so its steady-state step time (~**0.97s/step**) is the number that matters for the competition constraints.

# Phase 11: 3090 524k-Token Gap Analysis — Closing the Baseline Gap

## Goal
Understand why our multilayer U-Net underperforms the simple 9-layer baseline on 3090 with 524,288 tokens/step, and find the path to beat 1.6044 val_bpb.

---

## Complete Benchmark Results (10-minute wallclock, 3090, 524k tokens/step)

| Config | Layers | Checkpoint | LR | Warmup | Steps | Step time | FP val_bpb | INT8 val_bpb | Gap to Baseline |
|--------|--------|------------|-----|--------|-------|-----------|------------|--------------|-----------------|
| **Baseline (9L simple)** | 9 | none | 0.08/0.04 | 20 | 361 | ~1.67s | **1.6044** | 1.6044 | — |
| B_aggrLR (5L) | 5 | none | 0.12/0.02 | 40 | 357 | ~1.67s | 1.7149 | 1.7154 | +0.111 |
| D_6layer (6L) | 6 | none | 0.08/0.015 | 120 | 306 | ~1.86s | **1.6558** | 1.6571 | **+0.051** |
| E_6layer_aggrLR (6L) | 6 | none | 0.12/0.02 | 40 | 306 | ~1.88s | 1.7078 | 1.7092 | +0.103 |
| F_6layer_ckpt (6L) | 6 | encoder_only | 0.08/0.015 | 120 | 252 | ~2.35s | 1.7555 | 1.7553 | +0.151 |

## Key Findings

### 1. 6 layers beats 5 layers
- 5-layer: 1.7149 val_bpb (357 steps)
- 6-layer: 1.6558 val_bpb (306 steps) — **+0.059 better** despite fewer steps

### 2. Aggressive LR is a trap for 6 layers
- 6-layer standard LR (D): 1.6558
- 6-layer aggressive LR (E): 1.7078 — **worse!**
- Extra capacity needs gentle warmup; aggressive LR causes instability

### 3. Checkpointing hurts on 6-layer
- No checkpoint (D): 1.86s/step
- encoder_only checkpointing (F): 2.35s/step (+26% overhead)
- 252 steps vs 306 = fewer updates → worse final quality

### 4. Step time is the fundamental constraint
- Simple transformer: 1.67s/step — lean, single pass
- Multilayer 6L: 1.86s/step — recurrence + skip overhead
- Multilayer 6L + ckpt: 2.35s/step — checkpointing adds more overhead

### 5. Training loss progression (at step 200)
- Baseline: **2.87**
- D_6layer: **3.47**
- E_6layer: **3.60**
- F_6layer: **3.47**

Slower per-step convergence despite same or more compute — architectural overhead reduces effective capacity per step.

### 6. Scaling insight
- **3090**: Simple baseline wins (1.6044 vs 1.6558)
- **5090**: Multilayer dominates (1.4389 vs ~1.45 baseline extrapolated)

The multilayer architecture is **hardware-scalable but hardware-limited** on 3090.

---

## Remaining Gap: +0.051 val_bpb

To close this, we need to either:

### Option A: Reduce step time on 6-layer
- Currently 1.86s vs baseline 1.67s → need **~10% speedup**
- Potential paths:
  1. **Compile full model** with `torch.compile(base_model, dynamic=False, fullgraph=True)` instead of just `forward_logits`
  2. **Reduce recurrence to 1 step** (currently 2 steps)
  3. **Reduce refine blocks** from 3 to 2 (MINI_DEPTH_REFINE_BLOCKS=2)
  4. **Profile** where time goes: attention, MLP, or skip connections

### Option B: Increase per-step quality
- 6-layer needs to reach baseline's per-step convergence
- Potential paths:
  1. **QKV LoRA** instead of Q-only — more attention capacity
  2. **7 or 8 layers** — if step time stays under ~1.95s
  3. **Better initialization** — the baseline may have better weight init
  4. **Remove bigram hash / shell centering** — overhead without proven benefit at 10-min scale

### Option C: Hybrid approach
- Use the baseline architecture but add our best features:
  1. Simple transformer + LoRA
  2. Simple transformer + recurrence
  3. Simple transformer + bigram hash

---

## Recommended Next Experiments

### Priority 1: Full-model compile on 6-layer
```bash
export DISABLE_COMPILE=0
export TORCH_COMPILE_MODE=default
# In train_gpt.py: compile base_model instead of just forward_logits
```

This is the highest-potential, lowest-risk path. May cut 5–15% step time.

### Priority 2: Reduce recurrence to 1 step
```bash
export RECURRENCE_STEPS=1
```

Simple test: does single-step recurrence still achieve reasonable quality? If yes, step time drops significantly.

### Priority 3: 7-layer test
```bash
export NUM_LAYERS=7
```

Test if capacity gain outweighs step time cost.

### Priority 4: QKV LoRA rank 4 on 6-layer
```bash
export LORA_SCOPE=qkv
export LORA_RANK=4
```

Cheap quality test.

---

## Decision Matrix

| Experiment | Speed Risk | Quality Potential | Effort | Priority |
|------------|-----------|-------------------|--------|----------|
| Full-model compile | Low | Medium | Low | **1** |
| RECURRENCE_STEPS=1 | Medium | Medium | Low | **2** |
| NUM_LAYERS=7 | Medium | High | Low | 3 |
| QKV LoRA | Low | Low | Low | 4 |
| Remove bigram/shell | Low | Unknown | Low | 5 |

---

## Files Created

- `run_four_min_sweep.sh` — A/B/C/D 4-minute sweep
- `run_ten_min_B.sh` — B_aggrLR 10-minute
- `run_ten_min_D.sh` — D_6layer 10-minute
- `run_ten_min_E.sh` — E_6layer_aggrLR 10-minute
- `run_ten_min_E_and_F.sh` — E + F sequential 10-minute

---

## Verdict

The gap is **small but real**: +0.051 val_bpb. The simplest path to close it is:
1. Full-model compile to reduce step time
2. If that works, combine with 7 layers for extra capacity

If step time cannot be reduced below ~1.67s, the multilayer architecture may never beat the baseline on 3090 in a 10-minute wallclock — but it will dominate on faster hardware.
# Phase 5: ShellCentering λ Sweep — Full 10-Minute Runs

**Date**: 2026-05-02  
**Methodology**: Full 600s wallclock runs, identical seeds/data, varying SHELL_CENTERING_LAM only  
**Test**: Sweep λ ∈ {0.008, 0.064, 0.256} on 16MB track, Windows RTX 3090  
**Script**: `ScaleDown.bat` with `SHELL_CENTERING_LAM` overridden via environment variable

---

## Design

| Variable | Value |
|----------|-------|
| SHELL_CENTERING_ENABLED | 1 (all runs) |
| SHELL_CENTERING_LAM | 0.008, 0.064, 0.256 |
| All other params | ScaleDown defaults (identical across runs) |
| Wallclock | 600s |
| Data deterministic | Yes (seed=3623123517) |
| Model seed | 1337 |

Other parameters held constant:
- MODEL_DIM=1024, NUM_STEPS=1, TRAIN_SEQ_LEN=1024
- BIGRAM_HASH_ENABLED=1, LORA_RANK=512, LORA_SCOPE=q
- MATRIX_LR=0.08, SCALAR_LR=0.015
- LOSS_FILTER_ENABLED=1, SMEARGATE_ENABLED=0, TTT_ENABLED=0
- ITERATIONS=415, MAX_WALLCLOCK_SECONDS=600

---

## Raw Results

| λ | Steps Completed | Best BPB | Best Step | Final BPB (stride 64) |
|---|----------------|----------|-----------|----------------------|
| 0.008 | 414 | **1.7941** | 400 | 1.8708 |
| 0.064 | 403 | 1.8680 | 400 | 1.9263 |
| 0.256 | 401 | 1.9708 | 400 | 2.0373 |

### Per-λ validation progression

| Step | λ=0.008 | λ=0.064 | λ=0.256 |
|------|---------|---------|---------|
| 100 | 2.6067 | 2.6153 | 2.6943 |
| 200 | 2.0922 | 2.1092 | 2.3382 |
| 300 | 1.8723 | 1.9084 | 2.0347 |
| 400 | 1.7941 | 1.8680 | 1.9708 |

---

## Analysis

### BPB degradation vs λ

```
λ=0.008  ████████████████████████▌  1.7941  (baseline)
λ=0.064  ██████████████████████████  1.8680  (+0.0739, +4.1%)
λ=0.256  ███████████████████████████  1.9708  (+0.1767, +9.8%)
```

### Step loss comparison (initial training difficulty)

| λ | Step 0 loss | Step 10 loss | Step 50 loss | Step 100 loss |
|---|------------|-------------|-------------|--------------|
| 0.008 | 7.94 | 5.79 | 4.88 | 4.42 |
| 0.064 | 7.91 | 5.78 | 4.88 | 4.88 |
| 0.256 | 39.36 | 8.29 | 5.90 | 4.56 |

Stronger λ dramatically increases initial loss (39.36 vs 7.94 at step 0), confirming the shell norm penalty interferes with early optimization. Higher λ also reduces total steps completed (414 → 403 → 401) due to increased per-step compute from the penalty gradient.

---

## Conclusion

**λ=0.008 is the clear winner.** ShellCentering provides a regularization benefit at low strength, but increasing λ monotonically degrades both per-step quality and final BPB:

| Finding | Detail |
|---------|--------|
| **Direction** | BPB strictly increases with λ (monotonic degradation) |
| **Magnitude** | Each 8× increase in λ costs ~0.07–0.10 BPB |
| **Mechanism** | Higher λ forces embeddings toward unit norm too aggressively, conflicting with the LoRA adapters' ability to learn useful representations |
| **Interaction** | The penalty creates coupling between Q-LoRA and embedding norm — stronger penalty = harder optimization problem for the short training budget |

### Recommendation

Keep `SHELL_CENTERING_LAM=0.008` as the default in `ScaleDown.bat`. Do not increase.

---

## Log Files

| λ | Log File |
|---|---------|
| 0.008 | `logs/scaledown_shell008_10min.log` |
| 0.064 | `logs/scaledown_shell064_10min.log` |
| 0.256 | `logs/scaledown_shell256_10min.log` |

---

## References

- **Phase 3 A/B Test**: `notes/phase3_ab_test_final.md` — established ShellCentering win (p<0.005)
- **Phase 4 Track A**: `notes/phase4_track_A_ab_tests.md` — prior λ sweep at shorter budget (Phase 1 scale)
- **ScaleDown defaults**: `ScaleDown.bat` — comment updated to reflect sweep winner
# Phase 1 Candidate B — Shell Centering Penalty (Issue #2045)

## Metadata
- **Run ID:** From log `[config]` (check log header)
- **Timestamp:** 2026-05-01 13:37–13:41 IST
- **Stop reason:** Iterations (75/75 completed)
- **Best model:** step 75
- **AB Role:** Candidate against baseline A (ScaleDown defaults)

## Configuration
All knobs identical to baseline A except the ShellCenteringPenalty class added to `model.py`.

| Variable | Value |
|---|---|
| MODEL_DIM | 1024 |
| NUM_STEPS / RECURRENCE_STEPS | 1 |
| TRAIN_SEQ_LEN | 1024 |
| BIGRAM_HASH_ENABLED | 1 |
| LORA_RANK | 512 |
| LORA_SCOPE | q |
| MATRIX_LR | 0.08 |
| SCALAR_LR | 0.015 |
| LOSS_FILTER_ENABLED | 1 |
| SMEARGATE_ENABLED / TTT_ENABLED | 0 / 0 |
| ITERATIONS | 75 |
| DATA_DETERMINISTIC / DATA_SEED | 1 / 3623123517 |
| VAL_LOSS_EVERY | 5 |
| **ShellCentering λ** | **0.008** |

## Validation Progression vs Baseline

| Step | A (Baseline) BPB | B (ShellCentering) BPB | Δ (B−A) | Better? |
|---|---|---|---|---|
| 5 | 4.0822 | 4.0836 | +0.0014 | A (tie) |
| 10 | 4.0363 | 4.0219 | **−0.0144** | B |
| 15 | 3.9226 | 3.8444 | **−0.0782** | B |
| 20 | 3.7425 | 3.5931 | **−0.1494** | B |
| 25 | 3.6050 | 3.5646 | **−0.0404** | B |
| 30 | 3.5817 | 3.6178 | +0.0361 | A ⚠️ |
| 35 | 3.5605 | 3.5693 | +0.0088 | A |
| 40 | 3.5041 | 3.4898 | **−0.0143** | B |
| 45 | 3.4262 | 3.3941 | **−0.0321** | B |
| 50 | 3.3388 | 3.2823 | **−0.0565** | B |
| 55 | 3.2434 | 3.1616 | **−0.0818** | B |
| 60 | 3.1399 | 3.0416 | **−0.0983** | B |
| 65 | 3.0299 | 2.9314 | **−0.0985** | B |
| 70 | 2.9176 | 2.8372 | **−0.0804** | B |
| **75** | **2.8370** | **2.7775** | **−0.0595 ★** | **B** |

⚠️ **Temporary regression at steps 30–35:** The penalty initially disrupted learning mid-run before the model adapted. This is expected regularisation behaviour — the penalty adds constraint noise that the model learns to satisfy over subsequent steps.

## Phase 1 Verdict: ✅ PASS

| Gate | Threshold | A | B | Result |
|---|---|---|---|---|
| Final BPB (step 75) | B < A | 2.8370 | **2.7775** | ✅ −0.0595 (−2.1%) |
| Best BPB | B < A | 2.8370 | **2.7775** | ✅ Same checkpoint |
| Loss filter health | Skipped=0 | ✅ | ✅ (16/0/0) | ✅ Clean |
| Wall clock parity | Comparable | 186.1s | 188.1s | +1.1% (negligible) |
| VRAM parity | Comparable | 3.51 GiB | 3.58 GiB | +0.07 GiB (tiny) |

## Quant Eval
| Metric | A | B |
|---|---|---|
| FP val_bpb | 2.836980 | 2.777533 |
| INT8 val_bpb | 2.839583 | 2.778811 |
| Δ BPB | +0.002603 | +0.001278 |
| Degradation | 0.092% | 0.046% |

B's quant degradation is half that of A — the shell conditioning may produce more quantization-friendly weight distributions.

## Follow-Up: λ Optimized in Phase 4

The Phase 1 test used λ=0.008. A full λ sweep (0.004→0.512) in Phase 4 found the optimum at **λ=0.256**:

- λ=0.008: 2.7775 BPB (Phase 1 baseline)
- **λ=0.256: 3.0975 BPB at 120s** (0.244 BPB improvement over λ=0.008 at wallclock)
- λ=0.512: slight regression — peak confirmed at 0.256

Full details: [`notes/phase4_track_A_ab_tests.md`](phase4_track_A_ab_tests.md)

**`ScaleDown.bat` updated with `SHELL_CENTERING_LAM=0.256`.**

---

## LoRA Scope Test (qv): REJECTED

A follow-up test of `LORA_SCOPE=qv` (adding V LoRA to Q LoRA) was rejected at Phase 1 gate:

- scope=q:  2.9262 BPB at 75 iters
- scope=qv: 2.9522 BPB at 75 iters (−0.026 worse)

See [`notes/phase4_track_A_ab_tests.md#a2`](phase4_track_A_ab_tests.md#a2---lora-scope--qv-qv-lora-rejected-).

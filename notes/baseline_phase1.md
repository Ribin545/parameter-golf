# Phase 1 Baseline (A) — Default ScaleDown Configuration

## Metadata
- **Run ID:** `0061c706-bdd7-4831-ab4d-ee4d7b2d95bc`
- **Timestamp:** 2026-05-01 13:22–13:27 IST
- **Stop reason:** Iterations (75/75 completed)
- **Best model:** step 75
- **Seed:** 1337 (model) | 3623123517 (data, deterministic)
- **AB Role:** Baseline reference point — all Phase 2 variants measured against this

## Configuration (Environment Variables)
| Variable | Value |
|---|---|
| MODEL_DIM | 1024 |
| NUM_STEPS | 1 |
| RECURRENCE_STEPS | 1 |
| TRAIN_SEQ_LEN | 1024 |
| BIGRAM_HASH_ENABLED | 1 |
| LORA_RANK | 512 |
| LORA_SCOPE | q |
| MATRIX_LR | 0.08 |
| SCALAR_LR | 0.015 |
| LOSS_FILTER_ENABLED | 1 |
| SMEARGATE_ENABLED | 0 |
| TTT_ENABLED | 0 |
| ITERATIONS | 75 |
| MAX_WALLCLOCK_SECONDS | 9999 |
| DATA_DETERMINISTIC | 1 |
| DATA_SEED | 3623123517 |
| VAL_LOSS_EVERY | 5 |
| DATA_PATH | `../../../data/datasets/fineweb10B_sp1024` |
| TOKENIZER_PATH | `../../../data/tokenizers/fineweb_1024_bpe.model` |

### Defaults (not explicitly overridden)
| Variable | Default |
|---|---|
| MLP_MULT | 5 |
| NUM_HEADS | 8 |
| NUM_KV_HEADS | 4 |
| VOCAB_SIZE | 1024 |
| TIE_EMBEDDINGS | 1 |
| ROPE_BASE | 10000.0 |
| LOGIT_SOFTCAP | 10.0 |
| QK_GAIN_INIT | 1.5 |
| LORA_LR | 0.015 (inherits SCALAR_LR) |
| CONTROL_LR | 0.015 |
| HEAD_LR | 0.008 |
| TIED_EMBED_LR | 0.06 |
| MUON_MOMENTUM | 0.95 |
| BETA1/BETA2 | 0.9/0.95 |
| GRAD_CLIP_NORM | 1.0 |
| SEQ_LEN_CURRICULUM | 0 |
| RECURRENCE_CURRICULUM | 0 |

## Validation Progression (Full 75 Steps)
| Step | Val Loss | Val BPB | Δ BPB | Train Time (ms) |
|---|---|---|---|---|
| 5 | 6.9147 | 4.0822 | — | 11,825 |
| 10 | 6.8369 | 4.0363 | -0.0459 | 18,742 |
| 15 | 6.6443 | 3.9226 | -0.1137 | 25,699 |
| 20 | 6.3393 | 3.7425 | -0.1801 | 32,395 |
| 25 | 6.1064 | 3.6050 | -0.1375 | 39,274 |
| 30 | 6.0669 | 3.5817 | -0.0233 | 46,192 |
| 35 | 6.0310 | 3.5605 | -0.0212 | 53,078 |
| 40 | 5.9355 | 3.5041 | -0.0564 | 59,799 |
| 45 | 5.8035 | 3.4262 | -0.0779 | 66,668 |
| 50 | 5.6554 | 3.3388 | -0.0874 | 73,625 |
| 55 | 5.4938 | 3.2434 | -0.0954 | 80,639 |
| 60 | 5.3185 | 3.1399 | -0.1035 | 87,477 |
| 65 | 5.1322 | 3.0299 | -0.1100 | 94,365 |
| 70 | 4.9419 | 2.9176 | -0.1123 | 101,303 |
| **75** | **4.8407** | **2.8370** ★ | **-0.0806** | **108,255** |

★ Final checkpoint. Every validation step (5→75) set a new best — model was still improving at step 75.

## Best Checkpoint
| Metric | Value |
|---|---|
| Best step | 75 |
| Best val_loss | 4.8407 |
| Best val_bpb | **2.8370** |
| best_model.pt | 34.01 MiB (16-bit) |
| best_model.int8.ptz | 8.14 MiB (int8+zlib, level 9) |
| Total params | 35,484,354 |

## Loss Filter Health
| Metric | Value |
|---|---|
| Accepted | 16 |
| Skipped | 0 |
| Fallback accepts | 0 |
| Verdict | ✅ Clean — no pathological batches; filter not interfering with training |


## Quant Eval (FP vs INT8)
| Metric | Value |
|---|---|
| FP val_bpb | 2.836980 |
| INT8 val_bpb | 2.839583 |
| Δ BPB | +0.002603 |
| Degradation | +0.092% |

## Final Export
| Metric | Value |
|---|---|
| final_model.pt | 34.01 MiB |
| final_model.int8.ptz | 8.14 MiB |

## Resource
| Metric | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| VRAM allocated (peak) | 3.51 GiB |
| VRAM reserved | 3.68 GiB |
| Training steps completed | 75 / 75 |
| Wall clock | 186.1s (~3.1 min) |
| Per-step avg | ~1,443ms (including validation at every 5th step) |

## Crash Fix Applied
The original Phase 1 run crashed at step 35 due to Windows file-locking on `torch.save("best_model.pt")`. Fix applied to `train_gpt.py`:

- **Line 457 (best checkpoint):** Replaced direct `torch.save(sd, "best_model.pt")` with delete-then-move pattern:
  ```python
  torch.save(sd, "best_model.pt.tmp")
  import shutil
  try:
      os.remove("best_model.pt")
  except FileNotFoundError:
      pass
  shutil.move("best_model.pt.tmp", "best_model.pt")
  ```
- **Line 595 (final export):** Same atomic pattern applied preventatively.

## Verdict
✅ **Baseline established.** Default `ScaleDown.bat` configuration converges to **val_bpb=2.8370** at step 75 in 186.1s on RTX 3090. Model was still improving at final step (no plateau). All Phase 2 candidate variants should be measured against this baseline using identical 75-iteration, deterministic-data protocol for apples-to-apples comparison.
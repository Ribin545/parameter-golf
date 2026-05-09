# Test Plan: 3090 — Bridge from 1.86 BPB to ~1.3 BPB
## Goal: Keep 672ms/step advantage while stacking capacity via depth, LoRA, and features

### Reference Point
| Config | DIM | Steps | Step Time | Best BPB |
|--------|-----|-------|-----------|----------|
| Phase 7 Winner | 1024 | 415 | 1500ms | 1.8399 |
| Config B (speed gate) | 512 | 867 | 672ms | 1.8632 |
| **5090 Target** | — | — | — | **~1.3** |

### Strategy: Depth × Width × Features on 512-dim speed chassis
The 512-dim base delivers 672ms/step. Each experiment adds one capacity vector:
- **Depth**: RECURRENCE_STEPS (2, 4, 6, 12) — tokens get multiple reasoning passes
- **Width**: LORA_RANK (256, 512) + LORA_SCOPE (full, qv) — more trainable capacity
- **Features**: TTT_ENABLED — per-token online adaptation
- **LR optimization**: Different dims/recurrence depths need different LRs

---

## Gate 1 Tests (75 iters, ~2 min each)
*All use: DISABLE_COMPILE=1, LOSS_FILTER=0, VAL_LOSS_EVERY=25, WARMUP_STEPS=4*
*Baseline for comparison: Config B (512-dim, 1-step, rank-128, scope=q) = 2.7277 BPB @ 75*

### Round 1: Scale LoRA on 512-dim (capacity via width)
| Test | RECURRENCE_STEPS | LORA_RANK | LORA_SCOPE | MLP_MULT | Key Hypothesis |
|------|-----------------|-----------|------------|----------|----------------|
| **H1** | 1 | 256 | q | 8 | Double LoRA capacity, minimal speed cost |
| **H2** | 1 | 512 | q | 8 | Max LoRA on q-only, test ceiling |
| **H3** | 1 | 256 | qv | 8 | Add value LoRA, test attention coverage |
| **H4** | 1 | 256 | full | 8 | Full attention + MLP LoRA, max width |
| **H5** | 1 | 256 | full | 5 | Full LoRA + lean MLP, better balance? |

### Round 2: Add Recurrence Depth (capacity via depth)
| Test | RECURRENCE_STEPS | LORA_RANK | LORA_SCOPE | MLP_MULT | Key Hypothesis |
|------|-----------------|-----------|------------|----------|----------------|
| **H6** | 2 | 128 | q | 8 | Depth=2 gives 2× thinking per token |
| **H7** | 4 | 128 | q | 8 | Depth=4, test diminishing returns |
| **H8** | 6 | 128 | q | 8 | Depth=6, pushing depth limit |
| **H9** | 2 | 256 | full | 5 | Depth + Width combo, balanced |
| **H10** | 4 | 256 | full | 5 | Aggressive depth + width |

### Round 3: Feature Toggles (capacity via adaptation)
| Test | TTT_ENABLED | Other Change | Key Hypothesis |
|------|-------------|---------------|----------------|
| **H11** | 1 | base 512 config | TTT adds per-token adaptation |
| **H12** | 1 | LORA_RANK=256, scope=full | TTT + max width |

### Round 4: Middle-Dim Sweet Spot
| Test | MODEL_DIM | RECURRENCE_STEPS | LORA_RANK | MLP_MULT | Key Hypothesis |
|------|-----------|-----------------|-----------|----------|----------------|
| **H13** | 640 | 2 | 256 | 6 | 640-dim balances speed vs capacity |
| **H14** | 768 | 2 | 256 | 5 | 768-dim, depth=2, test scaling |
| **H15** | 768 | 4 | 384 | 5 | 768-dim + depth + big LoRA |

---

## Gate 2: Promoted candidates get 240s budget

---

## Gate 3: Winner gets multi-seed + 10-min final

---

### Execution Log
| Test | Run ID | Best BPB @ 75 | vs Baseline | Status | Notes |
|------|--------|---------------|-------------|--------|-------|
| H1 | — | — | — | PENDING | — |
| H2 | — | — | — | PENDING | — |
| H3 | — | — | — | PENDING | — |
| H4 | — | — | — | PENDING | — |
| H5 | — | — | — | PENDING | — |
| H6 | — | — | — | PENDING | — |
| H7 | — | — | — | PENDING | — |
| H8 | — | — | — | PENDING | — |
| H9 | — | — | — | PENDING | — |
| H10 | — | — | — | PENDING | — |
| H11 | — | — | — | PENDING | — |
| H12 | — | — | — | PENDING | — |
| H13 | — | — | — | PENDING | — |
| H14 | — | — | — | PENDING | — |
| H15 | — | — | — | PENDING | — |
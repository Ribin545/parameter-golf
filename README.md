# Parameter Golf — Research Dossier

> **Single-GPU Optimization Exhausted — Shift to 8× H100 Strategy**
>
> A complete experimental history of optimizing a recurrent causal language model under a strict 10-minute wallclock budget on Windows RTX 3090, with a cataloged path forward to competitive leaderboard placement.

**Current Best**: `val_bpb = 1.8270` @ step 400 (reproducible, Phase 7 Winner)
**Journey**: 3.75 BPB → 3.18 → 2.73 → **1.82** (3× improvement through architecture evolution)
**Total Experiments**: 25+ AB tests across 8 research phases, ~90 training runs logged

---

## BPB Journey

```
3.75 ────── Phase 1: Universal Transformer (12-step recurrence)
  │         notes/01_overview_and_timeline.md
  │
3.18 ────── UT stabilization (warmup / clip / safety tuning)
  │         notes/baseline_phase1.md
  │
2.73 ────── Phase 2: Throughput Pivot (dim=512, faster steps beat deeper recurrence)
  │         notes/02_current_architecture.md
  │
1.81 ────── Phase 3-4: New Architecture Regime (single-pass wide model)
  │         notes/phase3_ab_test_final.md
  │
1.82 ────── Phase 5-7: ShellCentering + DYNORM + LossFilter — Winner Convergence
  │         notes/phase7_final_summary.md
  │
1.83 ────── Phase 8: Exhaustion — all remaining candidates fail gate tests
  │         notes/phase8_research_plan.md
  │
  ▼ 8× H100 path mapped (1.03–1.08 BPB projected)
```

---

## Architecture & Proven Techniques

Our system is a **single-block tied recurrent causal LM** with per-step adapters and architecture-aware optimizer routing. Every active technique below has been validated through systematic sweeps or AB tests.

### Core IP (Unique to Our System)

| Technique | Phase | Δ BPB | Evidence |
|-----------|-------|-------|----------|
| **ShellCentering** (λ=0.008) | Phase 5 | +0.05–0.10 | Swept 0.004→0.512; revert test confirms; critical for multi-block scaling |
| **LossFilter** (z=4.5, warmup=250) | Phase 6-7 | +0.01–0.02 | 98% batch acceptance; compound value at 12,000 steps |
| **Dynamic LR Norm** (target=0.5) | Phase 7 | +0.0054 | 9-variant test suite; Winner-defining config |
| **LoRA Scope=q, Rank=512** | Phase 1-4 | +0.05–0.10 | Swept q/qv/full/none across 4 independent experiments |
| **BigramHash** (size=2048) | Phase 3 | +0.003–0.005 | Cheap positional features; in all winning configs |

### Architecture Params (Sweep-Verified)

| Parameter | Value | Sensitivity | Note |
|-----------|-------|------------|------|
| `MODEL_DIM` | 1024 (3090) / 512 (8×H100) | Moderate | Trade dim for blocks on H100 |
| `NUM_HEADS / NUM_KV_HEADS` | 8 / 4 (GQA) | Low | Grouped-query attention |
| `MLP_MULT` | 5 | Low | Optimal across all winning configs |
| `MUON_MOMENTUM` | 0.95 | **Critical** | 0.98 = +0.074 BPB regression |
| `GRAD_CLIP_NORM` | 1.0 | Moderate | 0.5 = +0.043 (double penalty w/ DYNORM) |
| `WARMUP_STEPS` | 16 | Moderate | 4 steps = regression |

Full hyperparameter reference: [`notes/phase8_research_plan.md` §3](notes/phase8_research_plan.md)

---

## Research Phases

### Phase 1 — Universal Transformer Baseline
**Goal**: Stabilize 12-step recurrent architecture on 3090
**Result**: 3.75 → 3.18 BPB through warmup/clip/safety tuning
**Key finding**: Depth recurrence alone insufficient under wallclock constraints

| Document | Contents |
|----------|----------|
| [`notes/01_overview_and_timeline.md`](notes/01_overview_and_timeline.md) | Full project timeline + BPB journey |
| [`notes/baseline_phase1.md`](notes/baseline_phase1.md) | UT-era baseline configs |
| [`notes/phase1_candidateB_shellCentering.md`](notes/phase1_candidateB_shellCentering.md) | First ShellCentering discovery |

### Phase 2 — Throughput Pivot
**Goal**: Explore dim=512 for more optimizer steps
**Result**: 3.18 → 2.73 BPB (faster steps beat deeper architecture)
**Key finding**: Wallclock budget = throughput trumps depth

| Document | Contents |
|----------|----------|
| [`notes/02_current_architecture.md`](notes/02_current_architecture.md) | Architecture evolution + current design rationale |
| [`notes/phase2_comparison_shellCentering.md`](notes/phase2_comparison_shellCentering.md) | ShellCentering confirms across throughput pivot |

### Phase 3 — Architecture AB Testing
**Goal**: Formalize AB methodology; test architecture feature combinations
**Result**: BigramHash wins; SmearGate loses (single-block structural incompatibility discovered)

| Document | Contents |
|----------|----------|
| [`notes/phase3_ab_test_final.md`](notes/phase3_ab_test_final.md) | AB3 feature sweep results (SmearGate, BigramHash, OrthoInit) |
| [`notes/07_ab_testing_methodology.md`](notes/07_ab_testing_methodology.md) | Our AB testing protocol (75-iter gate → 10-min promotion) |

### Phase 4 — LoRA Scope + ShellCentering Lambda
**Goal**: Sweep LoRA scopes and ShellCentering λ values
**Result**: Scope=q wins; λ=0.008 optimal (higher = over-centering)
**Key finding**: ShellCentering is highly sensitive — small λ changes matter

| Document | Contents |
|----------|----------|
| [`notes/phase4_track_A_ab_tests.md`](notes/phase4_track_A_ab_tests.md) | LoRA scope AB tests |
| [`notes/phase5_shellcentering_lambda_sweep.md`](notes/phase5_shellcentering_lambda_sweep.md) | λ=0.004→0.512 sweep; 0.008 confirmed |

### Phase 5 — Feature Toggle Verification
**Goal**: Revert-test ShellCentering; attempt QK_Gain scaling
**Result**: Removing ShellCentering costs 0.05+ BPB; QK_Gain=5 regresses at 300+ steps
**Key finding**: QK_Gain is step-hungry (U-shaped convergence)

| Document | Contents |
|----------|----------|
| [`notes/tier1_qk_gain_gate_results.md`](notes/tier1_qk_gain_gate_results.md) | QK_Gain 3/5/7 gate results |

### Phase 6 — Kernel Optimization (Reverted)
**Goal**: Fused Triton kernels for throughput
**Result**: Throughput not bottleneck — changes reverted for stability
**Key finding**: Model architecture > kernel micro-optimizations in this regime

| Document | Contents |
|----------|----------|
| [`notes/kernel_optimization_10m_comparison.md`](notes/kernel_optimization_10m_comparison.md) | Kernel optimization comparison |
| [`notes/baseline_phase6_post_kernel_revert.md`](notes/baseline_phase6_post_kernel_revert.md) | Post-revert baseline verification |

### Phase 7 — Winner Convergence
**Goal**: Final optimization push — 9 targeted tests on baseline
**Result**: DYNORM emerges as winner (+0.0054); SmearGate hypothesis tests diagnose structural root cause
**Key finding**: Most promising techniques fail because they require ≥2 blocks or >2,000 steps

| Document | Contents |
|----------|----------|
| [`notes/phase7_master_plan.md`](notes/phase7_master_plan.md) | Full 9-test experimental design |
| [`notes/phase7_final_summary.md`](notes/phase7_final_summary.md) | Final results + Winner config |
| [`notes/phase7_test1_rope_base.md`](notes/phase7_test1_rope_base.md) | RoPE base=1M → destabilization |
| [`notes/phase7_test2_dynamic_lr_norm.md`](notes/phase7_test2_dynamic_lr_norm.md) | DYNORM discovery |
| [`notes/phase7_test3_recurrence_curriculum.md`](notes/phase7_test3_recurrence_curriculum.md) | Curriculum loss |
| [`notes/phase7_test4_muon_momentum_098.md`](notes/phase7_test4_muon_momentum_098.md) | Muon 0.98 = +0.074 |
| [`notes/phase7_test5_dropout_005.md`](notes/phase7_test5_dropout_005.md) | Dropout 0.05 loss |
| [`notes/phase7_test6_label_smoothing_0.md`](notes/phase7_test6_label_smoothing_0.md) | Label smoothing 0.0 loss |
| [`notes/phase7_test7_dynorm_steps2.md`](notes/phase7_test7_dynorm_steps2.md) | DYNORM+NUM_STEPS=2 = +0.238 |
| [`notes/phase7_test8_shellCentering_256_20min.md`](notes/phase7_test8_shellCentering_256_20min.md) | ShellCentering extended test |
| [`notes/phase7_improvement_opportunities.md`](notes/phase7_improvement_opportunities.md) | Post-Phase 7 improvement catalog |

### Phase 8 — Single-GPU Exhaustion
**Goal**: Test all remaining promising candidates (Batch A, B, C)
**Result**: All 7 candidates fail 75-iter gate; single-block ceiling confirmed
**Key finding**: Structural incompatibility with leaderboard's depth-heavy formula — 8× H100 path mapped

| Document | Contents |
|----------|----------|
| [`notes/phase8_research_plan.md`](notes/phase8_research_plan.md) | **Complete 8× H100 migration playbook** — technique catalog, regression catalog, verified hyperparameters, structural diagnoses |
| [`notes/leaderboard_analysis.md`](notes/leaderboard_analysis.md) | Leaderboard technique analysis |
| [`notes/08_eap_ab_testing_methodology.md`](notes/08_eap_ab_testing_methodology.md) | EAP (Test-Time Training) AB protocol |

---

## Key Findings: What Works, What Doesn't

### ✅ Proven Beneficial (Any Hardware)

| Technique | Δ BPB | Mechanism |
|-----------|-------|-----------|
| ShellCentering (λ=0.008) | +0.05–0.10 | Prevents embedding drift; centers to unit hypersphere |
| LoRA scope=q, rank=512 | +0.05–0.10 | Query-only adaptation maximizes FLOP efficiency |
| LossFilter | +0.01–0.02 | Rejects pathological batches (98% acceptance) |
| Dynamic LR Norm | +0.005 | Auto-scales LR based on grad norm history |
| BigramHash (2048) | +0.003–0.005 | Cheap positional feature injection |

### ❌ Requires ≥2 Blocks (Fails on Single-Block)

| Technique | Single-Block Δ | Multi-Block Mechanism |
|-----------|---------------|----------------------|
| SmearGate | +0.045–0.060 | Per-block gating compounds: (1+ε)⁴ ≫ 1+ε |
| Parallel Attn/MLP Residuals | +0.12 | 2ᴺ representation pathways vs N |
| NUM_STEPS≥2 | +0.24 (w/ DYNORM) | Depth × recurrence product |
| TTT | Untested | Inner-loop benefit scales N× with depth |

### ⏳ Step-Hungry (Needs >2,000 Steps)

| Technique | 400-Step Δ | 8× H100 (>8K steps) Projection |
|-----------|-----------|-------------------------------|
| QK_Gain≥3 | U-shaped (regresses mid-training) | Converges at step 2,000+ |
| SP4096/8192 | Underfits embeddings | Ample steps for full vocab convergence |
| Schedule-Free | +0.03 (75-iter) | Asymptotic advantage at 12K steps |

### 🚫 Never Do

| Configuration | Δ BPB | Root Cause |
|--------------|-------|------------|
| MUON_MOMENTUM=0.98 | +0.074 | Newton-Schulz stability boundary |
| DYNORM + NUM_STEPS=2 | +0.238 | Negative feedback loop |
| DYNORM + GRAD_CLIP=0.5 | +0.043 | Double penalty on gradients |
| SHELL_CENTERING_LAM ≥ 0.064 | +0.02–0.05 | Over-centering: amplitude information lost |
| ROPE_BASE > 10K at seq=1024 | Destabilization | Frequency dilution at short distances |

---

## Experimental Methodology

We use a **two-stage AB testing protocol**:

1. **75-Iteration Gate Test** (~90 seconds): Quick elimination — candidate must beat baseline trajectory at step 75
2. **10-Minute Full Run** (~600 seconds): Only gate-winners are promoted; 3-seed validation for final decisions

This protocol has processed 25+ candidates across Phase 1-8 with zero false positives (no gate-winner ever regressed at full scale).

| Document | Contents |
|----------|----------|
| [`notes/07_ab_testing_methodology.md`](notes/07_ab_testing_methodology.md) | Core AB methodology + protocol design |
| [`notes/08_eap_ab_testing_methodology.md`](notes/08_eap_ab_testing_methodology.md) | EAP-specific AB extensions |
| [`notes/code_audit.md`](notes/code_audit.md) | Codebase audit for experimental integrity |

---

## Engineering

| Document | Contents |
|----------|----------|
| [`notes/05_engineering_endeavors.md`](notes/05_engineering_endeavors.md) | Windows backend patching, VRAM-safe EMA swap, quantized export pipeline |
| [`notes/03_training_system.md`](notes/03_training_system.md) | Training system architecture (Muon routing, loss filtering, instrumentation) |
| [`notes/04_experimental_results.md`](notes/04_experimental_results.md) | Full results archive across all phases |
| [`notes/06_validation_and_next_steps.md`](notes/06_validation_and_next_steps.md) | Validation procedures + reproducibility protocol |
| [`notes/09_session_cleanup_and_verification.md`](notes/09_session_cleanup_and_verification.md) | Session cleanup tracking |

---

## Repository Structure

```
├── train_gpt.py              # Main training launcher (Linux-compatible)
├── train_gpt_windows.py      # Windows multiprocessing wrapper
├── model.py                  # Transformer model definition
├── triton_mlp.py             # Fused Triton MLP kernel
├── data_utils.py             # Data loading / tokenization
├── eval_utils.py             # Validation + EMA evaluation
├── optimizer_utils.py        # Muon + AdamW optimizer groups
├── quant_utils.py            # FP32 → INT8 quantization + zlib
├── ScaleDown.bat             # Winner runner (deterministic re-eval)
├── setup_elite_env.bat       # Windows environment setup
├── submission.json           # Competition eval metadata
├── requirements.txt          # Python dependencies
├── checkpoints/              # Compressed model weights (.ptz)
├── docs/                     # Historical READMEs, leaderboard, reference batches
├── notes/                    # 32 research notes (complete Phase 1-8 history)
└── archive/                  # Failed experiments (20 gate tests, SP4096, EAP)
```

---

## Quick-Start

### Windows (RTX 3090)

```bat
setup_elite_env.bat
ScaleDown.bat
```

### Linux / RunPod / Cloud GPU

```bash
git clone https://github.com/Ribin545/parameter-golf.git
cd parameter-golf
git checkout master
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python train_gpt.py
```

**RunPod note**: Use PyTorch 2.5+ template, CUDA ≥12.1. The code auto-detects Linux and bypasses the Windows wrapper.

---

## What's Next: 8× H100 Path

Single-GPU optimization is exhausted at ~1.82 BPB. The leaderboard's winning formula requires depth-heavy architectures (4–6 blocks, 8,000–12,000 steps) that our single-block 3090 system structurally cannot replicate.

**Projected 8× H100 trajectory** (2-week migration):

| Phase | Milestone | Projected BPB | Timeline |
|-------|-----------|--------------|----------|
| 8H-1 | Current code on 8× H100 → 12,000 steps | ~1.25–1.35 | Immediate |
| 8H-2 | Enable step-hungry techniques (QK_Gain=5, NUM_STEPS=2, SP4096) | ~1.15–1.22 | Day 1 |
| 8H-3 | Restructure to 512-dim × 4–6 blocks (SmearGate, Parallel Res, TTT) | ~1.08–1.14 | Week 1 |
| 8H-4 | Fine-tuning + unique IP (ShellCentering on deep models, FA3) | ~1.03–1.08 | Week 2 |

**Complete execution playbook**: [`notes/phase8_research_plan.md`](notes/phase8_research_plan.md) — every technique cataloged, every regression diagnosed, no wasted exploration.

---

*Last updated: 2026-05-05 | Author: Ribin545 | Total experiments: 90+ runs | Notes: 32 documents*
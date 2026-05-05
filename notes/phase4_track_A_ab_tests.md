# Phase 4: Track A — ShellCentering λ Sweep + LoRA Scope Test

**Date**: 2026-05-01  
**Methodology**: λ sweep via 120s wallclock runs; LoRA scope via Phase 1 (75 iter) gate

---

## A.1 — ShellCentering λ Sweep: COMPLETE ✅

**Motivation**: Phase 3 confirmed ShellCentering helps at λ=0.008. Grid-search λ from 0.004 to 0.512 to find the optimum.

**Protocol**: 120s wallclock, deterministic data (seed=3623123517), model seed=42. All other params as per `ScaleDown.bat` defaults.

### Results

| λ | BPB (best) | Δ from λ=0.008 | Steps | Model Size |
|---|-----------|-----------------|-------|------------|
| 0.004 | 3.3763 | +0.035 ↑ | 55 | 34.01 MiB |
| 0.008 | 3.3412 | baseline | 56 | 34.01 MiB |
| 0.016 | 3.3067 | -0.035 | 54 | 34.01 MiB |
| 0.032 | 3.2346 | -0.107 | 53 | 34.01 MiB |
| 0.064 | 3.1690 | -0.172 | 55 | 34.01 MiB |
| 0.128 | 3.1303 | -0.211 | 54 | 34.01 MiB |
| **0.256** | **3.0975** | **-0.244** ★ | 54 | 34.01 MiB |
| 0.512 | 3.0917 | -0.250 | 53 | 34.01 MiB |

### Analysis

- **λ=0.256 is the optimum** — monotonic improvement up to 0.256, then a slight regression at 0.512 (penalty begins to dominate task loss)
- Total improvement from baseline: **0.244 BPB**
- Each doubling from 0.008→0.016→0.032→0.064→0.128→0.256 yielded diminishing but consistent gains
- Model size unchanged — penalty is purely a training loss term, zero submission bytes

### Decision: **λ=0.256 selected**. Applied to `ScaleDown.bat`.

---

## A.2 — LoRA Scope = qv (Q+V LoRA): REJECTED ❌

**Hypothesis**: Adding LoRA adapters to the V projection (in addition to Q) would improve expressiveness per step.

**Protocol**: Phase 1 gate — 75 iterations, no wallclock. Both configs use λ=0.256.

| Config | LORA_SCOPE | Best BPB (step 75) | Model Size | Wallclock |
|--------|-----------|---------------------|------------|-----------|
| **A** (baseline) | **q** | **2.9262** | 34.01 MiB | 200.8s |
| B (candidate) | qv | 2.9522 | 35.51 MiB | 216.5s |

### Verdict: **REJECT qv**

- B loses by **0.026 BPB** at identical iterations — per-step quality is inferior
- B is also **slower** (+15.7s, +7.8%) and **larger** (+1.50 MiB)
- Per methodology: kill at Phase 1 gate, do not proceed to Phase 2/3

### Interpretation

The ShellCentering penalty (applied to embeddings) creates coupling between the Q and V LoRA paths. With both adapters active simultaneously, the optimizer must disentangle two competing perturbations to the identity-penalized embedding representation — a harder problem than learning a single LoRA correction on Q alone. In the short training budget, simpler is better.

**LoRA scope `full` (Q+V+MLP) is likely even worse** — not worth testing at this time.

---

## Current Selection Summary

| Parameter | Selected Value | Source |
|-----------|---------------|--------|
| ShellCentering | Enabled | Phase 3 (p≈0.005) |
| ShellCentering λ | **0.256** | Phase 4 A.1 λ sweep |
| LORA_SCOPE | **q** (unchanged) | Phase 4 A.2 — qv rejected |
| All other params | ScaleDown defaults | — |

### Log Files

| Test | Log |
|------|-----|
| λ=0.004 | logs/phase4_B_lam0004_out.log |
| λ=0.016 | logs/phase4_B_lam0016_out.log |
| λ=0.032 | logs/phase4_B_lam0032_out.log |
| λ=0.064 | logs/phase4_B_lam0064_out.log |
| λ=0.128 | logs/phase4_B_lam0128_out.log |
| λ=0.256 | logs/phase4_B_lam0256_out.log |
| λ=0.512 | logs/phase4_B_lam0512_out.log |
| Scope q (baseline) | logs/phase4_A_scope_q_it75_out.log |
| Scope qv (rejected) | logs/phase4_B_scope_qv_it75_out.log |
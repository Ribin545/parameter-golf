# Leaderboard Analysis — What We Can Use

**Date**: 2026-05-02
**Source**: [parameter-golf leaderboard](https://github.com/openai/parameter-golf) as of April 2026

---

## Our Current State

| Metric | Value |
|--------|-------|
| Track | non-record-unlimited-compute-16mb |
| Val BPB (submitted) | 1.8614 |
| Best recent BPB (Phase 5) | 1.7941 |
| GPU | Single RTX 3090 (Windows) |
| Wallclock | 600s |
| Vocab | SP1024 (vocab_size=1024) |
| Model dim | 1024 |
| Unique features | ShellCentering, Depth-LoRA (LORA_SCOPE=q), LossFilter |

---

## Leaderboard Evolution & Technique Stack

The leaderboard progressed from ~1.22 BPB (baseline) to ~1.06 BPB (SOTA). Here are the techniques that drove improvements, ordered by approximate impact:

### Tier 1 — High-Impact, Likely Portable (worth testing on our system)

| Technique | First Seen | BPB Gain | Our Status | Feasibility |
|-----------|-----------|---------|------------|-------------|
| **SP8192 / SP4096 vocab** | PR #1218, #1394 | ~0.08-0.10 | We use SP1024 | **Easy** — env var change, larger vocab = more tokens in 16MB budget tradeoff |
| **QK-Gain 5.0-5.25** | PR #1413, #1493 | ~0.01-0.02 | We use 1.5 (conservative) | **Already tested** — 5.0 wins at 75-iters but regresses at 300+ iters; needs Phase 2/3 validation |
| **Parallel Residuals** | PR #1204, #1334, #1412 | ~0.01-0.02 | Not implemented | **Medium** — architecture change; parallel attention/MLP lane routing |
| **CaseOps tokenizer** | PR #1729, #1736 | ~0.005-0.01 | Not implemented | **Medium** — lossless bijective case transform; reduces vocab burden, more params for model |
| **SmearGate** | PR #198, #1667 | ~0.01-0.02 | DISABLED (causal leak) | **Fix the leak, re-enable** — this is high priority |

### Tier 2 — Medium-Impact, Gated by Throughput

| Technique | First Seen | BPB Gain | Our Status |
|-----------|-----------|---------|------------|
| **TTT (Test-Time Training)** | PR #549, #1514 | ~0.02-0.03 | We have a LoRA TTT path (eap_fit.py / 08_eap_ab_testing_methodology.md) |
| **Depth recurrence (layer looping)** | PR #1204, #1285, #1394 | ~0.01-0.015 | We have RECURRENCE_STEPS but at NUM_STEPS=1 |
| **VarLen attention / FA3** | PR #1530 | ~0.005-0.01 | We use fixed seq_len=1024 |
| **Fused MLP (Triton)** | PR #1530 | throughput | We have triton_mlp.py |

### Tier 3 — Optimization & Quantization (compression-side)

| Technique | First Seen | Effect | Our Status |
|-----------|-----------|--------|------------|
| **Muon variants** (MuonEq-R, Muon 0.97) | PR #1120, #1285 | ~0.005-0.01 | We use standard Muon with MUON_BACKEND_STEPS=5 |
| **Higher weight decay** (WD=0.08-0.09) | PR #1218, #1285 | ~0.005 | Our SCALAR_WD=0.1 (already high) |
| **GPTQ variants** (int6, GPTQ-lite, all-int6) | PR #287, #1019 | compression | We do int8 zlib; could try int6 |
| **XSA (extra self-attention)** | PR #198, #287, #1019 | ~0.005-0.01 | Not implemented |
| **MLPClip / SparseAttnGate** | PR #1769, #1787 | ~0.003-0.005 | Not implemented |
| **EMA replacing SWA** | PR #287, #374 | stability | Not implemented |
| **BigramHash sizing** (2048-10240) | multiple | ~0.003-0.005 | We use 2048 (can scale) |

---

## What Our Historical Data Says

From our journey (`notes/01_overview_and_timeline.md`, `logs/bpb_full_journey.csv`):

1. **Throughput >> depth strategy** — The single biggest insight: "More optimizer steps in wallclock beat slower depth strategy." This aligns with leaderboard's focus on fast kernels (FA3, fused MLP, Triton).

2. **ShellCentering is our unique win** — Phase 3 proved it (p<0.005 improvement). No leaderboard entry uses this technique. It synergizes with our LoRA-based architecture.

3. **QK-Gain 5.0 is a trap** — Wins at 75 iterations (Phase 1 scale) but regresses at 300+ iterations (full 600s runs). The leaderboard uses QK-Gain 5.0-5.25 on 8xH100 which completes far more steps — the dynamics differ.

4. **SmearGate leak must be fixed** — It's used by nearly every top-10 entry. Our SmearGate_BUG was due to non-causal leakage. Fixing this could unlock significant gains.

5. **LossFilter is also unique** — Another technique not seen on leaderboard. LOSS_FILTER_WARMUP=250, Z_THRESHOLD=4.5.

---

## Recommended Attack Vector (Single-GPU RTX 3090)

Given our constraint (RTX 3090, not 8xH100, no distributed training), the leaderboard's throughput-heavy techniques (VarLen FA3, fused Triton MLP, etc.) are less impactful for us. Focus on **per-step quality** improvements:

### Priority 1: Fix SmearGate
- Debug the causal leak in SmearGate implementation
- Re-run Phase 1 AB test (SmearGate+ShellCentering vs ShellCentering alone)
- Expected gain: ~0.01-0.02 BPB

### Priority 2: Test SP4096 vocab
- Change VOCAB_SIZE=4096, TOKENIZER_PATH to sp4096 tokenizer
- Tradeoff: smaller model (fewer params for same code size) but better compression per byte
- Run Phase 1 AB test at ITERATIONS=75

### Priority 3: Parallel residuals
- Implement parallel attention/MLP residual routing
- This is used by ~15 of the top 20 entries
- Phase 1 gate first

### Priority 4: Increase Muon momentum
- Test Muon beta=0.97 (from PR #1514 which showed p=0.020 improvement)
- Simple hyperparameter change

### Priority 5: CaseOps tokenizer
- If SP4096 shows promise, implement CaseOps for further vocab efficiency
- This is a code-size vs model-size optimization

---

## Techniques NOT Worth Pursuing (for our setup)

| Technique | Why Skip |
|-----------|---------|
| VarLen FA3 | Requires FA3 library, minimal gain on single GPU |
| Distributed all-reduces | Single GPU only |
| 8xH100 speed hacks | Our bottleneck is per-step quality, not throughput scaling |
| Ternary / 1-bit quantization | Too aggressive; quality loss outweighs compression gain for our score |
| Phased TTT | Complex, leaderboard shows it's stabilizing around legal score-first TTT |

---

## Log Files & References

| Resource | Path |
|----------|------|
| Journey CSV | `logs/bpb_full_journey.csv` |
| Phase 3 ShellCentering win | `notes/phase3_ab_test_final.md` (p<0.005) |
| Phase 5 λ sweep | `notes/phase5_shellcentering_lambda_sweep.md` |
| QK-Gain experiment | `notes/tier1_qk_gain_gate_results.md` |
| EAP TTT methodology | `notes/08_eap_ab_testing_methodology.md` |
| Our submission | `submission.json` (BPB 1.8614) |
| ScaleDown defaults | `ScaleDown.bat` |

---

## Summary

**Our best shot at improvement:** Fix SmearGate + test SP4096 vocab + ShellCentering (our unique advantage). These three are low-risk, high-potential, and don't conflict with each other. The leaderboard's top techniques (TTT, parallel residuals, QK-Gain 5.25, CaseOps) are all well-proven and worth staging after the priority items.
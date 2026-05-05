## 07 — AB Testing Methodology

This document defines the standard protocol for comparing two architecture/training configurations (A = baseline, B = candidate) under the 10-minute wallclock constraint on Windows RTX 3090.

The protocol is **architecture-agnostic**: any change to `model.py`, `train_gpt.py`, or training hyperparameters can be tested as long as the runtime entrypoint exposes the relevant knobs via environment variables.

---

### Design Philosophy

| Principle | Why |
|---|---|
| **Three-phase gating** | Kill bad candidates at the cheapest gate; invest budget only in promising ones |
| **Budget-aware** | Each phase is intentionally cheap (2–12 min total) so you can screen many ideas |
| **Reproducible** | Every phase pins deterministic data order and fixed seeds |
| **Log-driven** | All judgments come from parsing training logs — no real-time monitoring needed |

**Core rule:** If B cannot convincingly beat A at Phase 1, there is no reason to test it at Phase 2 or 3 — fix or discard.

---

### Phase 1 — Per-Step Quality Gate

**Purpose:** Is B a better architecture *per gradient step*? This strips out throughput effects entirely.

| Parameter | Value |
|---|---|
| `ITERATIONS` | 75 (identical for A and B) |
| `MAX_WALLCLOCK_SECONDS` | 9999 (wallclock disabled) |
| `DATA_DETERMINISTIC` | 1 |
| `DATA_SEED` | 3623123517 |
| All other knobs | Identical between A and B except the architectural change under test |

**Measurement:** Best checkpoint BPB (`val_bpb` from `[best]` log lines).

**Win condition:** B's best BPB < A's best BPB.

**Stop condition:** If B's best BPB >= A's best BPB, **kill the candidate** — it is not better per-step. Do not proceed.

**Expected cost:** ~2 minutes per config (~4 minutes total).

---

### Phase 2 — Proportional Budget Proxy

**Purpose:** Test combined architecture + throughput signal at 40% of the full wallclock budget.

| Parameter | Value |
|---|---|
| `MAX_WALLCLOCK_SECONDS` | 240 |
| `DATA_DETERMINISTIC` | 1 |
| `DATA_SEED` | 3623123517 |
| All other knobs | Identical between A and B except the architectural change |

**Measurements:**

1. Final stride-64 BPB (`[FINAL STRIDE 64]` line)
2. Total iterations completed (throughput signal)
3. Best checkpoint BPB (mid-run validation)

**Win condition:** B's final BPB < A's final BPB **and** B's best BPB < A's best BPB.

**Tie/draw:** If B wins on one metric but loses on the other, escalate to Phase 3 before making a decision.

**What to watch:** If B's architecture is heavier but its final BPB is lower, B is genuinely stronger per-unit-budget. If B is lighter but loses on final BPB, the throughput advantage isn't converting.

**Expected cost:** ~4 minutes per config (~8 minutes total).

---

### Phase 3 — Statistical Confidence (Multi-Seed)

**Purpose:** Confirm B's advantage is real and not seed noise.

| Parameter | Value |
|---|---|
| `MAX_WALLCLOCK_SECONDS` | 120 (per run) |
| `DATA_DETERMINISTIC` | 1 |
| `DATA_SEED` | 3623123517 (fixed data order), but vary `--seed` for model init |
| Seeds tested | 42, 1337, 2024 (or 3 distinct seeds of your choice) |
| Runs per config | 3 (one per seed) |

**Measurements:**

- Best checkpoint BPB for each seed
- Mean BPB across 3 seeds
- Standard deviation across 3 seeds
- Overlap check: B's mean + 1σ < A's mean − 1σ ?

**Win condition:** B's mean BPB < A's mean BPB **and** the ±1σ bands do not overlap (or overlap is minimal, ≤ 0.002 BPB).

**If bands overlap significantly:** The signal is within noise. Run more seeds or increase budget to 180s per run before making a call.

**Expected cost:** ~2 minutes per run × 6 runs = ~12 minutes total.

---

### Log Parsing Reference

All metrics come from the standard training log format. Extract these lines:

| Pattern | Metric | Phase |
|---|---|---|
| `[best] new_best ... val_bpb:X.XXXX` | Best checkpoint BPB | 1, 2, 3 |
| `[FINAL STRIDE 64]` | Final stride-64 BPB | 2 |
| `[stop] reason:iterations step:NNN` or `reason:wallclock step:NNN` | Total iterations completed | 2 |
| `[filter] totals accepted=... skipped=...` | Loss filter health check | All |
| `[config] run_id=...` | Run identification | All |

**Parsing algorithm for any phase:**

1. Locate the log file for config A run and config B run.
2. Grep all `[best] new_best` lines; take the **last one** (lowest BPB).
3. Grep `[FINAL STRIDE 64]` line; extract `val_bpb`.
4. Grep `[stop]` line; extract step count.
5. Compare numerically.

---

### Winner Selection Rubric

| Phase 1 | Phase 2 | Phase 3 | Verdict |
|---|---|---|---|
| B < A | — | — | Proceed to Phase 2 |
| B >= A | — | — | **REJECT B** — per-step quality insufficient |
| B < A | B < A on both metrics | B < A, no σ overlap | **ACCEPT B** — clear winner |
| B < A | B < A on one metric only | B < A, no σ overlap | **ACCEPT B** (cautious) — real but narrow |
| B < A | B < A on one metric only | Bands overlap | **TIE** — run full 10-minute comparison |
| B < A | B < A on both metrics | Bands overlap | **ACCEPT B** (cautious) — consistent at 240s, sample variance at 120s |

**Tie-breaking:**

- If the candidate introduces a new capability (e.g., recurrence depth, TTT, new feature path) and ties on BPB at Phase 2, prefer the candidate — it opens more future search dimensions.
- If the candidate simplifies the codebase and ties on BPB, prefer the candidate — lower maintenance cost.

**When to escalate to full 10-minute run:**

- Phase 3 shows B wins but with overlapping σ bands.
- B wins Phase 2 by > 0.02 BPB — run the full 10 minutes to get a merge-quality BPB number.

---

### Common Pitfalls

| Pitfall | Prevention |
|---|---|
| **SmearGate leakage** | Always set `SMEARGATE_ENABLED=0` unless the candidate explicitly tests a *fixed* smeargate path. Never compare a smeargate-enabled run against a non-smeargate baseline — causal leakage invalidates the comparison. |
| **TTT contamination** | Both A and B must have identical `TTT_ENABLED` and `TTT_LR` settings. TTT changes evaluation dynamics; mixing TTT and non-TTT configs produces non-comparable BPB values. |
| **Data seed mismatch** | All runs in the same phase must use identical `DATA_DETERMINISTIC=1` and `DATA_SEED`. Changing data order between A and B invalidates the comparison. |
| **VRAM OOM on B** | Architectural changes (higher dim, more heads, deeper recurrence) can exceed 24 GB VRAM. If B OOMs, reduce `MICRO_BATCH_TOKENS` or `TRAIN_BATCH_TOKENS` for B and note the throughput penalty in Phase 2 comparison. |
| **Loss filter divergence** | If loss filter stats (`skipped` or `fallback_accepts`) differ dramatically between A and B, the optimizer is behaving differently on the data — flag this and investigate before trusting the BPB numbers. |
| **Grad accum inconsistency** | Ensure `GRAD_ACCUM_STEPS` is recalculated or explicitly set identically for both configs. A mismatch changes the effective batch composition. |

---

### Quick Reference Card

```
GATE 1: ITERS=75, WALLCLOCK=9999, compare best BPB
         → B loses? KILL. B wins? → GATE 2.

GATE 2: WALLCLOCK=240, compare final + best BPB
         → B loses on both? KILL. Mixed? → GATE 3.

GATE 3: 3 seeds × 120s each, compare mean ± σ
         → B lower mean, bands don't overlap? ACCEPT.
         → Bands overlap? Escalate to full 10-min run.

TOTAL COST TO SCREEN ONE CANDIDATE: 4 + 8 + 12 = ~24 minutes max
                                     (but most candidates die at Gate 1 at ~4 min)
# Test Plan v2: Sub-600ms Target for 3090 → ~1.3 BPB
## Reality: 635ms is the compile-ON floor with all features active. Goal: trim compute.

### Verified Speed Benchmarks
| Config | DISABLE_COMPILE | Dim | Recur | MLP_MULT | Features | Step Time |
|--------|:-:|----:|-----:|-----:|------|------:|
| Config B | 1 (OFF) | 512 | 1 | 8 | All | 672ms |
| **C1** | **0 (ON)** | **512** | **1** | **8** | **All** | **635ms** |
| H6 | 1 | 512 | 2 | 8 | All | 1096ms |
| H7 | 1 | 512 | 4 | 8 | All | 2033ms |

### Compute Levers to Reduce Step Time
1. **MLP_MULT reduction**: 8→4 halves MLP compute (~50% of block time)
2. **Kill BigramHash**: embedding lookup + add every step
3. **Kill ShellCentering**: norm computations every step (minor but cumulative)
4. **Dropout reduction**: 0.15→0.05 or 0.00
5. **Label smoothing**: 0.05→0.00 (cheaper CE)
6. **Parallel residual**: fuse attn+mlp (reduces sequential depth)
7. **Fewer grad_accum_steps**: increase micro_batch_tokens? (VRAM tradeoff)

### Phase 1: Speed Trim (ITERATIONS=30, compile ON)
*Goal: Find config hitting sub-600ms, even if BPB is slightly worse.*

| Test | MLP_MULT | BigramHash | ShellCent | Dropout | LabelSm | Step Time | Notes |
|------|-----|------|-------|-----|-----|------|------|
| **C2** | 4 | ON | ON | 0.15 | 0.05 | ? | Half MLP, rest same |
| **C3** | 4 | OFF | ON | 0.15 | 0.05 | ? | -Bigram |
| **C4** | 4 | OFF | OFF | 0.05 | 0.00 | ? | Minimal features |
| **C5** | 4 | OFF | OFF | 0.00 | 0.00 | ? | Max speed trim |

### Phase 2: Best Speed Config → Gate 1 BPB Test (75 iters)
*Once we have sub-600ms, test BPB quality.*

### Phase 3: Capacity Stacking on Fast Chassis
*Try recurrence=2 ON the fast chassis (might be ~700-800ms)*

---

### Execution Log
| Test | Run ID | Dim | MLP_MULT | Bigram | ShellCent | Dropout | LabelSm | DT | BPB@30 | Status |
|------|--------|----:|-----|------|-------|-----|-----|----:|----:|--------|
| C1 | C1_compile_on_baseline | 512 | 8 | ON | ON | 0.15 | 0.05 | **635ms** | 3.448 | DONE |
| C2 | C2_mult4_baseline | 512 | 4 | ON | ON | 0.15 | 0.05 | **520ms** | 3.486 | DONE |
| C3 | C3_mult4_noBigram | 512 | 4 | OFF | ON | 0.15 | 0.05 | **511ms** | 3.495 | DONE |
| **C4** | C4_minimal | **512** | **4** | **OFF** | **OFF** | **0.15** | **0.05** | **493ms** | **3.491** | DONE |
| C4F | C4_full75 | 512 | 4 | OFF | OFF | 0.15 | 0.05 | **491ms** | **2.7823** | DONE |
| C5 | C5_mult6_full | 512 | 6 | ON | ON | 0.15 | 0.05 | **593ms** | **2.7476** | DONE |
| C6 | C6_mult6_parallel | 512 | 6 | ON | ON | 0.15 | 0.05 | **591ms** | **2.7452** | DONE |
| **C7** | **C7_minDrop_NoSm** | **512** | **6** | **ON** | **ON** | **0.05** | **0.00** | **581ms** | **2.7193** | **✓ WINNER** |

### AB Test Results (C7 vs Config B baseline)
| Gate | Config B (mlp_mult=8, d=0.1, ls=0.2) | C7 (mlp_mult=6, d=0.05, ls=0.0) | Verdict |
|------|------|------|------|
| Gate 1 (75 iters) | 2.7277 BPB, 672ms | **2.7193 BPB, 581ms** | C7 wins |
| Gate 2 Best (240s) | 1.9032 @350 | **1.9031** @350 (tie) | Tie |
| Gate 2 Final (240s) | **1.9518** @380 | 1.9598 @381 | D edge |
| 10-min Best | ~1.86xx | **1.8503** @400 | C7 wins |
| 10-min Final | ~1.93 | **1.9392** @998 | C7 wins |
| 10-min Int8 Size | ~4.43 MiB | **4.14 MiB** | C7 wins |

**Final decision: C7 config adopted.** Compile-ON vs OFF is a tie at 240s but compile-ON is slightly cheaper.

### Key Findings
- **torch.compile** saves only ~37ms (672→635) — Block already dominated by Triton MLP (dynamo-disabled)
- **MLP_MULT 8→4**: saves 115ms (635→520) — biggest single lever
- **BigramHash**: costs ~9ms when compiled — negligible, may be worth keeping
- **ShellCentering**: costs ~18ms — zero BPB benefit at step 30, likely droppable
- **Dropout 0.15**: hardcoded in model.py Block — only removable via code edit
- **Label smoothing 0.05**: hardcoded in GPT.forward — only removable via code edit
- C4 at 493ms leaves **107ms headroom** before 600ms ceiling for stacking recurrence

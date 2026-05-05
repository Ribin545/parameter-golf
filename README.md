# Parameter Golf — Training Stack (RTX 3090 / RunPod)

Single-block recurrent causal LM with ShellCentering, LossFilter, Dynamic LR Norm, and quantized export.
**Best reproducible: val_bpb=1.8270** @ step 400 (10-min wallclock).

## Quick-Start

### Windows (RTX 3090 — local dev)

```bat
# One-time setup
setup_elite_env.bat

# Run 10-min training + quantized export
ScaleDown.bat
```

### Linux / RunPod / Server Deployment

```bash
# 1. Clone
git clone <your-repo-url>
cd <repo>

# 2. Create venv + install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run training
python train_gpt.py
```

**RunPod note:** Use a PyTorch 2.5+ template with CUDA ≥12.1. The code auto-detects Linux and uses the native `train_gpt.py` path (no Windows wrapper needed).

### Environment Variables

All configuration is via env vars. See `ScaleDown.bat` for the full reference set. Key ones:

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL_DIM` | 1024 | 512 for multi-block on H100 |
| `MAX_WALLCLOCK_SECONDS` | 600 | 10-min competition limit |
| `SHELL_CENTERING_ENABLED` | 1 | Core IP — don't disable |
| `SHELL_CENTERING_LAM` | 0.008 | Validated optimal (Phase 5 sweep) |
| `LORA_SCOPE` | q | q > qv > full (Phase 1-4) |
| `LORA_RANK` | 512 | 512 > 256 > 128 |
| `DYNAMIC_LR_NORM` | 1 | +0.005 BPB (Phase 7 Winner) |
| `TARGET_GRAD_NORM` | 0.5 | Never combine DYNORM with NUM_STEPS>1 |
| `LOSS_FILTER_ENABLED` | 1 | 98% batch acceptance rate |

## File Structure

```
├── train_gpt.py              # Main training launcher
├── train_gpt_windows.py      # Windows multiprocessing wrapper
├── model.py                  # Transformer model definition
├── triton_mlp.py             # Fused Triton MLP kernel
├── data_utils.py             # Data loading / tokenization
├── eval_utils.py             # Validation + EMA evaluation
├── optimizer_utils.py        # Muon + AdamW optimizer groups
├── quant_utils.py            # FP32 → INT8 quantization + zlib export
├── ScaleDown.bat             # Winner runner (deterministic re-eval)
├── setup_elite_env.bat       # Windows environment setup
├── submission.json           # Competition eval metadata
├── requirements.txt          # Python dependencies
├── checkpoints/              # Compressed model weights (.ptz)
├── docs/                     # Historical READMEs, leaderboard, reference batches
├── notes/                    # Research notes (Phase 1-8, AB methodology)
└── archive/                  # Failed experiments (tests/, sp4096/, eap/)
    ├── tests/                # 20 failed Phase 7-8 gate tests
    ├── sp4096_experiments/   # SP4096 vocab experiments
    └── eap/                  # Test-time training exploration
```

## Research Notes

| Document | Contents |
|----------|----------|
| `notes/phase8_research_plan.md` | Complete 8× H100 migration playbook with technique catalog |
| `notes/07_ab_testing_methodology.md` | Our AB testing protocol |
| `notes/phase7_final_summary.md` | Phase 7 Winner results |
| `docs/README.md` | Original project README (archival) |

## What Never to Do

- **Never** combine `DYNORM=1` with `NUM_STEPS>1` (+0.238 BPB regression)
- **Never** set `MUON_MOMENTUM=0.98` (+0.074 BPB — optimizer instability)
- **Never** set `SHELL_CENTERING_LAM ≥ 0.064` (over-centering)
- **Never** use `ROPE_BASE > 10K` at seqlen=1024 (frequency dilution)

See `notes/phase8_research_plan.md` §4 for the complete regression catalog.
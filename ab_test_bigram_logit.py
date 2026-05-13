#!/usr/bin/env python3
"""
A/B Test: Bigram Logit Bias — Phase 10

Compares:
- Variant A (baseline): BIGRAM_LOGIT_ENABLED=0
- Variant B (Phase 10): BIGRAM_LOGIT_ENABLED=1

Both use the Phase 9c config otherwise identical.
Runs 75 steps (same as ab_test.py) for speed.
"""
from __future__ import annotations
import os, sys, subprocess, pathlib, re, time

ROOT = pathlib.Path(__file__).parent
LOG_DIR = ROOT / "ab_logs"
IS_WINDOWS = sys.platform == "win32"


def run_config(name: str, bigram_logit_enabled: int) -> pathlib.Path:
    """Run training with bigram logit on/off. Returns log path."""
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"bigram_logit_{name}.txt"

    env = os.environ.copy()
    env.update({
        # Phase 9c base config
        "MODEL_TYPE": "multilayer",
        "NUM_LAYERS": "5",
        "MODEL_DIM": "512",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2",
        "RECURRENCE_STEPS": "2",
        "MULTILAYER_LORA_RANK": "8",
        "MICRO_BATCH_TOKENS": "153600",
        "TRAIN_BATCH_TOKENS": "153600",
        "TRAIN_SEQ_LEN": "1024",
        "OPTIM_MODE": "muon_adam",
        "MATRIX_LR": "0.15",
        "SCALAR_LR": "0.04",
        "LORA_LR": "0.04",
        "CONTROL_LR": "0.04",
        "BETA2": "0.95",
        "DROPOUT_P": "0.30",
        "LABEL_SMOOTHING": "0.08",
        "SCHEDULE_FREE": "1",
        "WARMUP_STEPS": "40",
        "DYNAMIC_LR_NORM": "1",
        "TARGET_GRAD_NORM": "0.8",
        "RECURRENT_ATTN_EVERY": "2",
        "BIGRAM_HASH_ENABLED": "1",
        "BIGRAM_HASH_SIZE": "4096",
        "BIGRAM_HASH_SCALE": "0.05",
        "SHELL_CENTERING_ENABLED": "1",
        "SHELL_CENTERING_LAM": "0.005",
        "TIE_EMBEDDINGS": "0",
        "MLP_MEMORY_MODE": "checkpoint",
        "ATTN_MEMORY_MODE": "checkpoint",
        "MULTILAYER_ACTIVATION_CHECKPOINT": "1",
        "MULTILAYER_ACTIVATION_CHECKPOINT_MODE": "encoder",
        "MLP_RECOMPUTE": "1",
        "SDPA_BACKEND": "auto",
        # A/B test specific
        "ITERATIONS": "75",
        "MAX_WALLCLOCK_SECONDS": "9999",
        "DATA_DETERMINISTIC": "1",
        "DATA_SEED": "3623123517",
        "DISABLE_COMPILE": "1",
        "SAVE_BEST_CHECKPOINT": "0",
        "SAVE_BEST_INT8": "0",
        "EXPORT_BEST_CHECKPOINT": "0",
        "QUANT_EVAL": "0",
        "VAL_LOSS_EVERY": "10",
        "TRAIN_LOG_EVERY": "50",
        "VOCAB_SIZE": "1024",
        "SAFETY_CLAMP_DISABLE": "1",
        # The variable under test
        "BIGRAM_LOGIT_ENABLED": str(bigram_logit_enabled),
        "BIGRAM_LOGIT_SCALE_INIT": "0.05",
    })

    script = "train_gpt_windows.py" if IS_WINDOWS else "train_gpt.py"

    print(f"\n{'='*60}")
    print(f"Running Variant {name}: BIGRAM_LOGIT_ENABLED={bigram_logit_enabled}")
    print(f"{'='*60}")

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(ROOT),
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            if "val_bpb" in line or "step:" in line and "dt:" in line:
                print(f"  {line.rstrip()}")
        proc.wait()

    if proc.returncode != 0:
        print(f"[ERROR] Variant {name} failed with exit code {proc.returncode}")
    else:
        print(f"[OK] Variant {name} completed")

    return log_path


def parse_val_bpb(log_path: pathlib.Path) -> tuple[float | None, float | None]:
    """Extract final val_loss and val_bpb from log."""
    val_loss = None
    val_bpb = None
    with open(log_path, "r") as f:
        for line in f:
            # Match: step:75 val_loss:1.2345 val_bpb:1.5678 ...
            m = re.search(r"val_loss:([0-9.]+)\s+val_bpb:([0-9.]+)", line)
            if m:
                val_loss = float(m.group(1))
                val_bpb = float(m.group(2))
    return val_loss, val_bpb


def main():
    print("=" * 60)
    print("A/B Test: Bigram Logit Bias")
    print("=" * 60)

    # Variant A: Baseline (no bigram logit)
    log_a = run_config("A_baseline", bigram_logit_enabled=0)
    loss_a, bpb_a = parse_val_bpb(log_a)

    # Variant B: With bigram logit
    log_b = run_config("B_bigram_logit", bigram_logit_enabled=1)
    loss_b, bpb_b = parse_val_bpb(log_b)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if bpb_a is None or bpb_b is None:
        print("[ERROR] Could not parse val_bpb from one or both logs")
        print(f"  Log A: {log_a}")
        print(f"  Log B: {log_b}")
        sys.exit(1)

    delta_bpb = bpb_b - bpb_a
    delta_loss = loss_b - loss_a
    pct = (delta_bpb / bpb_a) * 100.0 if bpb_a > 0 else 0.0

    print(f"Variant A (baseline):  val_loss={loss_a:.4f}  val_bpb={bpb_a:.4f}")
    print(f"Variant B (bigram):    val_loss={loss_b:.4f}  val_bpb={bpb_b:.4f}")
    print(f"Delta:                 val_loss={delta_loss:+.4f}  val_bpb={delta_bpb:+.4f} ({pct:+.2f}%)")

    if delta_bpb < -0.005:
        print("\n✅ VERDICT: Bigram logit bias WINS (statistically significant improvement)")
    elif delta_bpb > 0.005:
        print("\n❌ VERDICT: Bigram logit bias LOSES (statistically significant degradation)")
    else:
        print("\n⚖️  VERDICT: TIE (difference within noise floor ±0.005)")

    # Also show step counts
    for name, log_path in [("A", log_a), ("B", log_b)]:
        with open(log_path, "r") as f:
            last_step = None
            for line in f:
                m = re.search(r"step:(\d+)\s+loss:", line)
                if m:
                    last_step = int(m.group(1))
            if last_step:
                print(f"  Variant {name} reached step {last_step}")


if __name__ == "__main__":
    main()
"""
AB Test Orchestrator - Phase 1: Per-Step Quality Gate

Runs Config A (recurrent 12-step) and Config B (11-layer multilayer).
Platform-aware: uses train_gpt_windows.py on Windows, train_gpt.py on Linux.
Parses logs and prints verdict.

Usage: python ab_test.py
"""
from __future__ import annotations
import os, sys, subprocess, pathlib, re, time

ROOT = pathlib.Path(__file__).parent
LOG_DIR = ROOT / "ab_logs"
IS_WINDOWS = sys.platform == "win32"


def run_config(name: str, model_type: str, extra_env: dict | None = None) -> pathlib.Path:
    """Run the training script for one config. Returns path to log file."""
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"config_{name.lower()}.txt"

    env = os.environ.copy()
    env.update({
        # Phase 1 params (per methodology)
        "ITERATIONS": "75",
        "MAX_WALLCLOCK_SECONDS": "9999",
        "DATA_DETERMINISTIC": "1",
        "DATA_SEED": "3623123517",
        "DATA_PATH": os.environ.get("DATA_PATH", "../../../data/datasets/fineweb10B_sp1024"),
        "TOKENIZER_PATH": os.environ.get("TOKENIZER_PATH", "../../../data/tokenizers/fineweb_1024_bpe.model"),
        # Model selection
        "MODEL_TYPE": model_type,
        "RECURRENCE_STEPS": "12",
        "NUM_LAYERS": "11",
        # Disable checkpoint/export overhead for AB test
        "SAVE_BEST_CHECKPOINT": "0",
        "SAVE_BEST_INT8": "0",
        "EXPORT_BEST_CHECKPOINT": "0",
        "QUANT_EVAL": "0",
        # Validate every 10 steps
        "VAL_LOSS_EVERY": "10",
    })
    if extra_env:
        env.update(extra_env)

    # Platform: Windows uses the compat wrapper; Linux uses train_gpt.py directly
    if IS_WINDOWS:
        launcher = str(ROOT / "train_gpt_windows.py")
    else:
        launcher = str(ROOT / "train_gpt.py")
        # Linux: let torch.compile use inductor + Triton natively
        env.pop("DISABLE_COMPILE", None)

    print(f"\n{'='*60}")
    print(f"[{name}] Starting run (MODEL_TYPE={model_type}, platform={sys.platform})...")
    print(f"[{name}] Launcher: {launcher}")
    print(f"[{name}] Log: {log_path}")
    print(f"{'='*60}")

    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            [sys.executable, "-u", launcher],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(ROOT), bufsize=1, universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"  [{name}] {line}", end="")
            log_f.write(line)
        proc.wait()
    return log_path


def parse_best_bpb(log_path: pathlib.Path) -> float | None:
    """Extract best val_bpb from [best] new_best log lines."""
    text = log_path.read_text(encoding="utf-8")
    pattern = re.compile(r"\[best\] new_best.*val_bpb:([\d.]+)")
    matches = pattern.findall(text)
    if not matches:
        return None
    return min(float(m) for m in matches)


def parse_final_bpb(log_path: pathlib.Path) -> float | None:
    """Extract final val_bpb from step line with [FINAL STRIDE 64]."""
    text = log_path.read_text(encoding="utf-8")
    pattern = re.compile(r"val_bpb:([\d.]+).*\[FINAL STRIDE 64\]")
    matches = pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def main():
    print("=" * 70)
    print("AB TEST - Phase 1: Per-Step Quality Gate")
    print("  Config A: Recurrent single-block x 12 steps (MODEL_TYPE=recurrent)")
    print("  Config B: 11 distinct layers  (MODEL_TYPE=multilayer)")
    print("  Protocol: 75 iters, deterministic data, compare best val_bpb")
    print("=" * 70)

    t_start = time.perf_counter()

    # Run Config A
    log_a = run_config("A", "recurrent")
    bpb_a = parse_best_bpb(log_a)

    # Free VRAM between runs
    import torch
    torch.cuda.empty_cache()

    # Run Config B
    log_b = run_config("B", "multilayer")
    bpb_b = parse_best_bpb(log_b)

    elapsed = time.perf_counter() - t_start

    # --- Results ---
    print(f"\n{'=' * 70}")
    print("RESULTS - Phase 1 Comparison")
    print(f"{'=' * 70}")
    print(f"  Total wallclock: {elapsed:.0f}s")
    print(f"  Config A (recurrent 1x12):  best_bpb = {bpb_a}")
    print(f"  Config B (11-layer):        best_bpb = {bpb_b}")

    if bpb_a is not None and bpb_b is not None:
        delta = bpb_b - bpb_a
        print(f"  Delta (B - A):              {delta:+.4f}")
        if bpb_b < bpb_a:
            print(f"\n  [WIN] B WINS - per-step quality is better.")
            print(f"        Proceed to Phase 2 (240s wallclock).")
        else:
            print(f"\n  [LOSS] B LOSES - per-step quality is worse or equal.")
            print(f"         KILL candidate.")
    else:
        print(f"\n  [ERROR] Could not parse BPB from one or both logs.")
        print(f"  Check {LOG_DIR}/config_a.txt and {LOG_DIR}/config_b.txt")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
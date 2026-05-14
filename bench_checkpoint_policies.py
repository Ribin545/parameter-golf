#!/usr/bin/env python3
"""
Benchmark selective checkpointing policies for GPTMultiLayer.
Compares step time, VRAM, and short-run val_bpb across policies.
"""

import os
import sys
import time
import json
import torch
import torch._inductor.config as inductor_config

# Always disable CUDA graphs (required for checkpointing safety)
inductor_config.triton.cudagraphs = False

# Base config
BASE_ENV = {
    "MLP_MEMORY_MODE": "checkpoint",
    "ATTN_MEMORY_MODE": "off",  # Block-level handles this
    "SDPA_BACKEND": "auto",
    "MULTILAYER_ACTIVATION_CHECKPOINT": "1",
    "MLP_RECOMPUTE": "1",
}

# Policies to test
POLICIES = [
    ("full", {}),
    ("attention_only", {}),
    ("late_attention", {}),
    ("no_refine_checkpoint", {}),
    ("no_mlp_checkpoint", {}),
]

DEVICE = torch.device("cuda")
BATCH = 150
SEQLEN = 1024
VOCAB = 1024
WARMUP_STEPS = 10
MEASURE_STEPS = 20
TOTAL_STEPS = 30  # For val_bpb

MODEL_ARGS = {
    "vocab_size": VOCAB,
    "num_layers": 5,
    "model_dim": 512,
    "num_heads": 8,
    "num_kv_heads": 4,
    "mlp_mult": 2,
    "tie_embeddings": False,
    "tied_embed_init_std": 0.005,
    "num_steps": 2,
    "logit_softcap": 30.0,
    "rope_base": 10000.0,
    "qk_gain_init": 3.0,
    "bigram_hash_size": 4096,
    "bigram_hash_scale": 0.05,
    "lora_rank": 8,
    "recurrent_attn_every": 2,
    "shell_centering_enabled": True,
    "shell_centering_lam": 0.005,
    "label_smoothing": 0.08,
    "z_loss_lambda": 0.0,
}


def reset_cuda():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def test_policy(policy_name: str, extra_env: dict) -> dict:
    """Test one checkpointing policy. Returns results dict."""
    # Set env
    env = {**BASE_ENV, "MULTILAYER_ACTIVATION_CHECKPOINT_MODE": policy_name, **extra_env}
    for k, v in env.items():
        os.environ[k] = str(v)

    # Must reimport to pick up env changes
    import importlib
    import model_multilayer
    importlib.reload(model_multilayer)
    from model_multilayer import GPTMultiLayer

    # Build model
    reset_cuda()
    model = GPTMultiLayer(**MODEL_ARGS).to(DEVICE).bfloat16().train()
    model.zero_grad()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora" in n)

    # Compile
    cm = torch.compile(model, mode="default")
    cm.train()

    # Inputs
    x = torch.randint(0, VOCAB, (BATCH, SEQLEN), device=DEVICE)
    y = torch.randint(0, VOCAB, (BATCH, SEQLEN), device=DEVICE)

    # Warmup
    for _ in range(WARMUP_STEPS):
        loss = cm(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(MEASURE_STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = cm(x, y)
        loss.backward()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        model.zero_grad(set_to_none=True)
        times.append(dt)

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_gb = peak_mb / 1024

    # Quick val_bpb (very short, just for sanity)
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for _ in range(3):
            logits = cm.forward_logits(x)
            val_loss += torch.nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB).float(), y.reshape(-1)
            ).item()
        val_bpb = (val_loss / 3) / 0.6931  # bits per byte approximation

    # Cleanup
    del model, cm, x, y
    reset_cuda()

    return {
        "policy": policy_name,
        "step_ms_avg": sum(times) / len(times),
        "step_ms_min": min(times),
        "step_ms_max": max(times),
        "vram_peak_gib": peak_gb,
        "total_params": total_params,
        "lora_params": lora_params,
        "val_bpb": val_bpb,
    }


def main():
    print("=" * 80)
    print("SELECTIVE CHECKPOINTING POLICY BENCHMARK")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Batch: {BATCH}, SeqLen: {SEQLEN}, Vocab: {VOCAB}")
    print(f"Model: {MODEL_ARGS}")
    print("-" * 80)

    results = []
    for policy, extra in POLICIES:
        print(f"\n>>> Testing policy: {policy}")
        try:
            r = test_policy(policy, extra)
            results.append(r)
            print(f"  Step time: {r['step_ms_avg']:.1f}ms "
                  f"(min={r['step_ms_min']:.1f}, max={r['step_ms_max']:.1f})")
            print(f"  VRAM peak: {r['vram_peak_gib']:.2f} GiB")
            print(f"  val_bpb:   {r['val_bpb']:.4f}")
            print(f"  Params:    {r['total_params']:,} total, {r['lora_params']:,} LoRA")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "policy": policy,
                "error": str(e),
            })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Policy':<20} {'Step (ms)':>10} {'VRAM (GiB)':>12} {'val_bpb':>10} {'Status':>10}")
    print("-" * 64)
    for r in results:
        if "error" in r:
            print(f"{r['policy']:<20} {'FAILED':>10} {'-':>12} {'-':>10} {'ERROR':>10}")
        else:
            status = "✅" if r["vram_peak_gib"] < 20 else "⚠️ HIGH"
            print(f"{r['policy']:<20} {r['step_ms_avg']:>10.1f} {r['vram_peak_gib']:>12.2f} "
                  f"{r['val_bpb']:>10.4f} {status:>10}")

    # Save results
    with open("checkpoint_policy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to checkpoint_policy_results.json")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Benchmark aggressive checkpointing policies to get under 600ms."""

import os, time, json, torch
import torch._inductor.config as inductor_config
inductor_config.triton.cudagraphs = False

BASE_ENV = {
    "MLP_MEMORY_MODE": "checkpoint",
    "ATTN_MEMORY_MODE": "off",
    "SDPA_BACKEND": "auto",
    "MULTILAYER_ACTIVATION_CHECKPOINT": "1",
    "MLP_RECOMPUTE": "1",
}

# Test ALL policies including new aggressive ones
POLICIES = [
    ("full", {}),
    ("no_mlp_checkpoint", {}),
    ("encoder_only", {}),
    ("minimal", {}),
    ("no_refine_checkpoint", {}),
    ("attention_only", {}),
]

DEVICE = torch.device("cuda")
BATCH, SEQLEN, VOCAB = 150, 1024, 1024
WARMUP, MEASURE = 10, 20

MODEL_ARGS = {
    "vocab_size": VOCAB, "num_layers": 5, "model_dim": 512,
    "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 2,
    "tie_embeddings": False, "tied_embed_init_std": 0.005,
    "num_steps": 2, "logit_softcap": 30.0, "rope_base": 10000.0,
    "qk_gain_init": 3.0, "bigram_hash_size": 4096,
    "bigram_hash_scale": 0.05, "lora_rank": 8,
    "recurrent_attn_every": 2, "shell_centering_enabled": True,
    "shell_centering_lam": 0.005, "label_smoothing": 0.08,
    "z_loss_lambda": 0.0,
}

def test_policy(name):
    env = {**BASE_ENV, "MULTILAYER_ACTIVATION_CHECKPOINT_MODE": name}
    for k, v in env.items():
        os.environ[k] = str(v)

    import importlib
    import model_multilayer
    importlib.reload(model_multilayer)
    from model_multilayer import GPTMultiLayer

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = GPTMultiLayer(**MODEL_ARGS).to(DEVICE).bfloat16().train()
    cm = torch.compile(model, mode="default")
    cm.train()

    x = torch.randint(0, VOCAB, (BATCH, SEQLEN), device=DEVICE)
    y = torch.randint(0, VOCAB, (BATCH, SEQLEN), device=DEVICE)

    for _ in range(WARMUP):
        cm(x, y).backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    times = []
    for _ in range(MEASURE):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cm(x, y).backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        model.zero_grad(set_to_none=True)

    peak = torch.cuda.max_memory_allocated() / (1024**3)

    del model, cm, x, y
    torch.cuda.empty_cache()

    return {
        "policy": name,
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "vram_gib": peak,
    }

print("=" * 70)
print("AGGRESSIVE CHECKPOINTING POLICY BENCHMARK")
print("=" * 70)
print(f"Target: <600ms  |  GPU: {torch.cuda.get_device_name(0)}")
print("-" * 70)

results = []
for policy in POLICIES:
    print(f"\n>>> {policy[0]}")
    try:
        r = test_policy(policy[0])
        results.append(r)
        status = "✅ UNDER 600!" if r["avg_ms"] < 600 else "⚠️ OVER 600"
        print(f"  {r['avg_ms']:.1f}ms  VRAM={r['vram_gib']:.2f}GiB  {status}")
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append({"policy": policy[0], "error": str(e)})

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Policy':<20} {'Step (ms)':>10} {'VRAM (GiB)':>12} {'Status':>15}")
print("-" * 60)
for r in results:
    if "error" in r:
        print(f"{r['policy']:<20} {'FAILED':>10} {'-':>12} {'ERROR':>15}")
    else:
        status = "🏆 UNDER 600!" if r["avg_ms"] < 600 else "Over 600"
        print(f"{r['policy']:<20} {r['avg_ms']:>10.1f} {r['vram_gib']:>12.2f} {status:>15}")

with open("aggressive_policy_results.json", "w") as f:
    json.dump(results, f, indent=2)
#!/usr/bin/env python3
"""Benchmark STATIC mini-depth: step 1 runs only last K blocks."""

import os, time, json, torch
import torch._inductor.config as inductor_config
inductor_config.triton.cudagraphs = False

BASE_ENV = {
    "MLP_MEMORY_MODE": "checkpoint",
    "ATTN_MEMORY_MODE": "off",
    "SDPA_BACKEND": "auto",
    "MULTILAYER_ACTIVATION_CHECKPOINT": "1",
    "MULTILAYER_ACTIVATION_CHECKPOINT_MODE": "encoder_only",
    "MLP_RECOMPUTE": "1",
    "DISABLE_COMPILE": "0",
    "TORCH_COMPILE_MODE": "default",
    "MINI_DEPTH_STATIC": "1",
}

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

def test_k(k):
    env = {**BASE_ENV, "MINI_DEPTH_REFINE_BLOCKS": str(k)}
    for key, val in env.items():
        os.environ[key] = str(val)

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
        loss = cm(x, y)
        loss.backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        model.zero_grad(set_to_none=True)

    peak = torch.cuda.max_memory_allocated() / (1024**3)

    # Quick val_bpb
    model.eval()
    with torch.no_grad():
        logits = cm.forward_logits(x)
        val_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB).float(), y.reshape(-1)
        ).item()
        val_bpb = val_loss / 0.6931

    del model, cm, x, y
    torch.cuda.empty_cache()

    return {
        "k": k,
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "vram_gib": peak,
        "val_bpb": val_bpb,
    }

print("=" * 70)
print("STATIC MINI-DEPTH: Step 1 runs only last K blocks")
print("=" * 70)
print(f"Full model: 5 blocks. Step 0 always runs all 5.")
print(f"Step 1 trimmed to last K blocks only.")
print("-" * 70)

configs = [5, 4, 3, 2, 1]
results = []
for k in configs:
    print(f"\n>>> K={k} (last {k} blocks in step 1)")
    try:
        r = test_k(k)
        results.append(r)
        status = "✅ FASTER" if r["avg_ms"] < 552 else "⚠️ same/slower"
        print(f"  {r['avg_ms']:.1f}ms  VRAM={r['vram_gib']:.2f}GiB  val_bpb={r['val_bpb']:.4f}  {status}")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append({"k": k, "error": str(e)})

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'K (blocks)':<12} {'Step (ms)':>10} {'VRAM (GiB)':>12} {'val_bpb':>10}")
print("-" * 55)
for r in results:
    if "error" in r:
        print(f"K={r['k']:<10} {'FAILED':>10} {'-':>12} {'-':>10}")
    else:
        print(f"K={r['k']:<10} {r['avg_ms']:>10.1f} {r['vram_gib']:>12.2f} {r['val_bpb']:>10.4f}")

with open("static_mini_depth_results.json", "w") as f:
    json.dump(results, f, indent=2)
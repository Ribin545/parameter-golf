"""Focused Phase 11 step-time audit for clean 6-layer GPTMultiLayer.

Tests performance-sensitive toggles without changing dims/recurrence/steps:
- BIGRAM_HASH_ENABLED: verifies disabled path removes embedding lookup/add
- ATTN_OUTPUT_MODE: baseline transpose+linear vs einsum_fused

This uses a single micro-batch (102400 tokens on 3090 clean config) and measures
forward+backward only. It is intended as a quick bug/perf regression audit, not
as a quality benchmark.
"""
from __future__ import annotations

import gc
import os
import statistics
import time

import torch


def reset_model_multilayer_globals() -> None:
    import model_multilayer as mm
    mm._MLP_MEMORY_MODE = None
    mm._ATTN_MEMORY_MODE = None
    mm._SDPA_BACKEND_MODE = None
    mm._ATTN_OUTPUT_MODE = None


def run_case(name: str, *, bigram: bool, attn_output_mode: str, compile_forward: bool = True) -> dict:
    os.environ.update({
        "BIGRAM_HASH_ENABLED": "1" if bigram else "0",
        "ATTN_OUTPUT_MODE": attn_output_mode,
        "SDPA_BACKEND": "flash",
        "MLP_MEMORY_MODE": "off",
        "ATTN_MEMORY_MODE": "off",
        "MULTILAYER_ACTIVATION_CHECKPOINT": "0",
        "MINI_DEPTH_STATIC": "1",
        "MINI_DEPTH_REFINE_BLOCKS": "3",
    })
    reset_model_multilayer_globals()
    from model_multilayer import GPTMultiLayer

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    gc.collect()

    model = GPTMultiLayer(
        vocab_size=1024,
        num_layers=6,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        num_steps=2,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        bigram_hash_size=4096,
        bigram_hash_scale=0.05,
        lora_rank=8,
        recurrent_attn_every=1,
        shell_centering_enabled=False,
        label_smoothing=0.0,
        z_loss_lambda=0.0,
    ).to(device).bfloat16().train()

    if compile_forward:
        model.forward_logits = torch.compile(model.forward_logits, mode="default")

    # Same as 3090 clamp in train_gpt.py: 102400 tokens = 100 seqs of 1024.
    bsz, seqlen = 100, 1024
    x = torch.randint(0, 1024, (bsz, seqlen), device=device)
    y = torch.randint(0, 1024, (bsz, seqlen), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    # Warmup includes compile/autotune cost; do not measure.
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(8):
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = model(x, y)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    result = {
        "name": name,
        "bigram": bigram,
        "attn_output_mode": attn_output_mode,
        "params": sum(p.numel() for p in model.parameters()),
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "last_loss": float(loss.detach().float()),
    }
    del model, opt, x, y, loss
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    cases = [
        ("clean_baseline_proj_no_bigram", False, "baseline"),
        ("clean_einsum_no_bigram", False, "einsum_fused"),
        ("old_einsum_with_bigram", True, "einsum_fused"),
    ]
    results = []
    for name, bigram, mode in cases:
        print(f"\n=== {name} ===", flush=True)
        res = run_case(name, bigram=bigram, attn_output_mode=mode)
        results.append(res)
        print(res, flush=True)

    print("\nSUMMARY")
    base = results[0]["median_ms"]
    for r in results:
        delta = r["median_ms"] - base
        pct = (delta / base) * 100.0
        print(f"{r['name']}: median={r['median_ms']:.2f}ms delta={delta:+.2f}ms ({pct:+.2f}%) params={r['params']:,}")


if __name__ == "__main__":
    main()
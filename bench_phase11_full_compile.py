"""Safe full-model compile benchmark for current winning 6-layer config.

Compares:
1) current partial compile (`forward_logits` only)
2) full-model compile (`torch.compile(model, ...)`)

Uses the Phase 11 winning recipe assumptions:
- 6 layers, 2 recurrence steps
- bigram off, shell centering off
- baseline attention output path
- refine_blocks=2
- no checkpointing

CUDA graphs are explicitly disabled for safety.
"""
from __future__ import annotations

import gc
import os
import statistics
import time

import torch
import torch._inductor.config as inductor_config

from model_multilayer import GPTMultiLayer


def make_model(device: torch.device) -> GPTMultiLayer:
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
    return model


def bench(label: str, full_compile: bool) -> dict:
    os.environ.update({
        "BIGRAM_HASH_ENABLED": "0",
        "SHELL_CENTERING_ENABLED": "0",
        "MINI_DEPTH_STATIC": "1",
        "MINI_DEPTH_REFINE_BLOCKS": "2",
        "ATTN_OUTPUT_MODE": "baseline",
        "SDPA_BACKEND": "flash",
        "MLP_MEMORY_MODE": "off",
        "ATTN_MEMORY_MODE": "off",
        "MULTILAYER_ACTIVATION_CHECKPOINT": "0",
        "FUSED_MLP_ENABLED": "1",
    })
    inductor_config.triton.cudagraphs = False

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    gc.collect()

    model = make_model(device)
    if full_compile:
        compiled = torch.compile(model, mode="default")
    else:
        model.forward_logits = torch.compile(model.forward_logits, mode="default")
        compiled = model

    bsz, seqlen = 100, 1024
    x = torch.randint(0, 1024, (bsz, seqlen), device=device)
    y = torch.randint(0, 1024, (bsz, seqlen), device=device)
    opt = torch.optim.AdamW(compiled.parameters(), lr=1e-4, fused=True)

    # Warmup
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        loss = compiled(x, y)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()

    times = []
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = compiled(x, y)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    result = {
        "label": label,
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "loss": float(loss.detach().float()),
    }
    del model, compiled, opt, x, y, loss
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

    partial = bench("partial_compile_forward_logits", full_compile=False)
    print(partial)
    full = bench("full_model_compile", full_compile=True)
    print(full)

    delta = full["median_ms"] - partial["median_ms"]
    pct = (delta / partial["median_ms"]) * 100.0
    print(f"SUMMARY partial={partial['median_ms']:.2f}ms full={full['median_ms']:.2f}ms delta={delta:+.2f}ms ({pct:+.2f}%)")


if __name__ == "__main__":
    main()
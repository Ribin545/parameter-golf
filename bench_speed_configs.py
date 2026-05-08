"""
Benchmark different model configurations for step time.
Tests 5L x2S multilayer variants with different dims, mlp_mults, heads.
"""
from __future__ import annotations
import os, time, torch
from model_multilayer import GPTMultiLayer

torch.cuda.init()
device = torch.device("cuda")

configs = [
    # (dim, heads, kv_heads, mlp_mult) - label
    (512, 4, 2, 8, "baseline"),
    (512, 4, 2, 4, "mlp4"),
    (512, 4, 2, 3, "mlp3"),
    (512, 4, 1, 4, "mlp4_kv1"),
    (448, 4, 2, 4, "dim448_mlp4"),
    (384, 4, 2, 4, "dim384_mlp4"),
    (384, 4, 2, 3, "dim384_mlp3"),
    (384, 4, 1, 3, "dim384_mlp3_kv1"),
    (320, 4, 2, 3, "dim320_mlp3"),
]

seq_len = 1024
micro_batch = 64  # 65536 tokens

for dim, heads, kv_heads, mlp_mult, label in configs:
    model = GPTMultiLayer(
        vocab_size=1024, num_layers=5, model_dim=dim,
        num_heads=heads, num_kv_heads=kv_heads, mlp_mult=mlp_mult,
        tie_embeddings=True, tied_embed_init_std=0.005,
        num_steps=2, bigram_hash_size=2048, bigram_hash_scale=0.05,
    ).to(device).bfloat16()
    
    total_p = sum(p.numel() for p in model.parameters())
    raw_mb = total_p * 2 / 1024 / 1024
    est_int8 = raw_mb * 0.24  # from actual compression ratios
    
    x = torch.randint(0, 1024, (micro_batch, seq_len), device=device)
    y = torch.randint(0, 1024, (micro_batch, seq_len), device=device)
    
    # Warmup
    for _ in range(3):
        loss = model(x, y)
        loss.backward()
        model.zero_grad()
    
    # Measure 5 steps
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        loss = model(x, y)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000 / 5
    
    # Also measure optimizer time on a separate pass
    loss = model(x, y)
    loss.backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Simulate AdamW step
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data -= 0.01 * p.grad
    torch.cuda.synchronize()
    opt_ms = (time.perf_counter() - t0) * 1000
    
    del model
    torch.cuda.empty_cache()
    
    print(f"{label:<20} dim={dim:>3} h={heads} kv={kv_heads} mlp={mlp_mult} | "
          f"fwd+bwd={dt_ms - opt_ms:.0f}ms opt={opt_ms:.0f}ms total={dt_ms:.0f}ms | "
          f"params={total_p/1e6:.1f}M raw={raw_mb:.0f}MiB int8~{est_int8:.1f}MiB {'✓' if est_int8<16 else '✗16MB'}")
"""
Quick profile: time breakdown of 5L x2S d=512 multilayer forward+backward.
Runs a single micro-batch to measure per-component costs.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import time
from model_multilayer import GPTMultiLayer

torch.cuda.init()
device = torch.device("cuda")

# Match target config: 5L, 2 steps, d=512
model = GPTMultiLayer(
    vocab_size=1024, num_layers=5, model_dim=512,
    num_heads=4, num_kv_heads=2, mlp_mult=8,
    tie_embeddings=True, tied_embed_init_std=0.005,
    num_steps=2, bigram_hash_size=2048, bigram_hash_scale=0.05,
).to(device).bfloat16()

seq_len = 256  # short sequence for profiling
batch = 32
x = torch.randint(0, 1024, (batch, seq_len), device=device)
y = torch.randint(0, 1024, (batch, seq_len), device=device)

# Warmup
for _ in range(5):
    model(x, y)

# Profile forward+backward
torch.cuda.synchronize()
t0 = time.perf_counter()
loss = model(x, y)
loss.backward()
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) * 1000
print(f"Full fwd+bwd (seq={seq_len}, batch={batch}): {dt:.1f}ms")

# Profile just forward
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    _ = model.forward_logits(x)
torch.cuda.synchronize()
dt_fwd = (time.perf_counter() - t0) * 1000
print(f"Forward only: {dt_fwd:.1f}ms")

# Profile block-level
x_emb = model.tok_emb(x).bfloat16()
x_emb = x_emb + model.bigram_hash(x)
x_emb = torch.nn.functional.rms_norm(x_emb, (x_emb.size(-1),))
x0 = x_emb.clone()

# Time a single block
block = model.blocks[0]
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    _ = block(x_emb, x0)
torch.cuda.synchronize()
dt_block = (time.perf_counter() - t0) * 1000 / 10
print(f"Single block fwd: {dt_block:.1f}ms")

# Time attention
attn = block.attn
z = block.attn_norm(x_emb)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    _ = attn(z)
torch.cuda.synchronize()
dt_attn = (time.perf_counter() - t0) * 1000 / 10
print(f"Attention only: {dt_attn:.1f}ms")

# Time MLP
mlp = block.mlp
z = block.mlp_norm(x_emb)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    _ = mlp(z)
torch.cuda.synchronize()
dt_mlp = (time.perf_counter() - t0) * 1000 / 10
print(f"MLP only: {dt_mlp:.1f}ms")

# Model size
total = sum(p.numel() for p in model.parameters())
raw_mb = total * 2 / 1024 / 1024
print(f"\nModel: {total:,} params, {raw_mb:.1f} MiB raw")

# Estimate training step cost (scale up to seq=1024, batch=64 tokens)
# FLOPs scale linearly with seq_len and batch
scale_factor = (1024 / seq_len) * (64 / batch)
est_step = dt * scale_factor
print(f"\nEstimated full training step (seq=1024, micro=64 tokens): ~{est_step/1000:.2f}s")
print(f"5 blocks × 2 passes = 10 block evaluations per step")
print(f"10 blocks × {dt_block:.1f}ms/block = {10*dt_block:.1f}ms fwd overhead")
import os
import sys
import time
import torch

os.environ["MLP_MEMORY_MODE"] = "checkpoint"
os.environ["ATTN_MEMORY_MODE"] = "checkpoint"
os.environ["SDPA_BACKEND"] = "math"
os.environ["ATTN_OUTPUT_MODE"] = "baseline"

sys.path.insert(0, "/mnt/e/Projects/Proj/golf/records/track_non_record_16mb/2026-04-01_EliteUTv22p8_12StepRecurrence_Windows3090")

from model_multilayer import GPTMultiLayer

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
B = 150
T = 1024
V = 1024

model = GPTMultiLayer(
    vocab_size=V, num_layers=5, model_dim=512, num_heads=8, num_kv_heads=4,
    mlp_mult=2, tie_embeddings=False, tied_embed_init_std=0.005, num_steps=2,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=3.0,
    bigram_hash_size=4096, bigram_hash_scale=0.05, lora_rank=8,
    recurrent_attn_every=2, shell_centering_enabled=True,
    shell_centering_lam=0.005, label_smoothing=0.08,
).to(device=DEVICE, dtype=DTYPE)

optim = torch.optim.AdamW(model.parameters(), lr=0.03, betas=(0.9,0.95), weight_decay=0.0)
model.train()

# Warmup (3 steps)
x = torch.randint(0, V, (B, T), device=DEVICE)
y = torch.randint(0, V, (B, T), device=DEVICE)
for _ in range(3):
    optim.zero_grad()
    loss = model(x, y)
    loss.backward()
    optim.step()
    torch.cuda.synchronize()

# Measure (5 steps)
times = []
for i in range(5):
    x = torch.randint(0, V, (B, T), device=DEVICE)
    y = torch.randint(0, V, (B, T), device=DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    optim.zero_grad()
    loss = model(x, y)
    loss.backward()
    optim.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    times.append(dt)
    print(f"step={i} dt={dt:.1f}ms loss={loss.item():.4f}", flush=True)

times = times[1:]  # drop first (JIT warmup)
median = sorted(times)[len(times)//2]
mean = sum(times)/len(times)
print(f"\nMedian={median:.1f}ms Mean={mean:.1f}ms Min={min(times):.1f}ms Max={max(times):.1f}ms", flush=True)
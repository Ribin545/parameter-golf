"""
Direct step-time benchmark for model_multilayer.py
Measures only forward+backward step time, no data loading overhead.
"""
import os
import time
import torch
from torch import nn
import torch.nn.functional as F

# Set env before importing model
os.environ["MLP_MEMORY_MODE"] = "off"
os.environ["ATTN_MEMORY_MODE"] = "off"
os.environ["SDPA_BACKEND"] = "math"
os.environ["ATTN_OUTPUT_MODE"] = "baseline"

from model_multilayer import GPTMultiLayer

# Config matching Phase 9b
VOCAB_SIZE = 1024
NUM_LAYERS = 5
MODEL_DIM = 512
NUM_HEADS = 8
NUM_KV_HEADS = 4
MLP_MULT = 2
RECURRENCE_STEPS = 2
LORA_RANK = 8
BATCH_TOKENS = 153600
SEQ_LEN = 1024
BATCH_SIZE = BATCH_TOKENS // SEQ_LEN  # 150

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

print(f"Device: {DEVICE}, dtype: {DTYPE}")
print(f"Batch: {BATCH_SIZE} seqs × {SEQ_LEN} tokens = {BATCH_TOKENS} tokens")

model = GPTMultiLayer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    model_dim=MODEL_DIM,
    num_heads=NUM_HEADS,
    num_kv_heads=NUM_KV_HEADS,
    mlp_mult=MLP_MULT,
    tie_embeddings=False,
    tied_embed_init_std=0.005,
    num_steps=RECURRENCE_STEPS,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=3.0,
    bigram_hash_size=4096,
    bigram_hash_scale=0.05,
    lora_rank=LORA_RANK,
    recurrent_attn_every=2,
    shell_centering_enabled=True,
    shell_centering_lam=0.005,
    label_smoothing=0.08,
    z_loss_lambda=0.0,
).to(device=DEVICE, dtype=DTYPE)

# Simple Adam optimizer (matches matrix + scalar groups)
optim = torch.optim.AdamW(
    model.parameters(),
    lr=0.03,
    betas=(0.9, 0.95),
    weight_decay=0.0,
)

model.train()

print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Warmup data
input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
target_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

# Warmup (10 steps)
print("Warmup...")
for _ in range(10):
    optim.zero_grad()
    loss = model(input_ids, target_ids)
    loss.backward()
    optim.step()
    torch.cuda.synchronize()

# Measure (10 steps)
print("Measuring...")
times = []
for i in range(10):
    # Fresh data to avoid cache effects
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    target_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    optim.zero_grad()
    loss = model(input_ids, target_ids)
    loss.backward()
    optim.step()
    
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000  # ms
    
    times.append(dt)
    print(f"step:{i} dt:{dt:.2f}ms loss:{loss.item():.4f}")

times = times[1:]  # drop first (JIT warmup)
print(f"\nMedian step time: {sorted(times)[len(times)//2]:.2f}ms")
print(f"Mean step time: {sum(times)/len(times):.2f}ms")
print(f"Min: {min(times):.2f}ms  Max: {max(times):.2f}ms")
import os, time, torch
import torch._inductor.config as inductor_config

# Disable CUDA graphs in compile (safe with checkpointing)
inductor_config.triton.cudagraphs = False

os.environ.update({
    "MLP_MEMORY_MODE": "checkpoint",
    "ATTN_MEMORY_MODE": "checkpoint",
    "MULTILAYER_ACTIVATION_CHECKPOINT": "1",
    "MULTILAYER_ACTIVATION_CHECKPOINT_MODE": "encoder",
    "MLP_RECOMPUTE": "1",
    "SDPA_BACKEND": "auto",
})

from model_multilayer import GPTMultiLayer

device = torch.device("cuda")
model = GPTMultiLayer(
    1024, 5, 512, 8, 4, 2, False, 0.005, 2, 30.0, 10000.0, 3.0,
    4096, 0.05, 8, 2, True, 0.005, 0.08, 0.0
).to(device).bfloat16()

# Compile the full model but WITHOUT CUDA graphs
cm = torch.compile(model, mode="default")

x = torch.randint(0, 1024, (150, 1024), device=device)
y = torch.randint(0, 1024, (150, 1024), device=device)

# Warmup
for _ in range(5):
    cm(x, y).backward()
    model.zero_grad(set_to_none=True)

# Test
times = []
torch.cuda.synchronize()
for i in range(20):
    t0 = time.perf_counter()
    loss = cm(x, y)
    loss.backward()
    torch.cuda.synchronize()
    dt = (time.perf_counter()-t0)*1000
    model.zero_grad(set_to_none=True)
    times.append(dt)
    if i < 5:
        print(f"step {i}: {dt:.1f}ms")

avg = sum(times)/len(times)
mn = min(times)
mx = max(times)
print(f"\nAvg: {avg:.1f}ms  Min: {mn:.1f}ms  Max: {mx:.1f}ms")
print(f"vs baseline ~700ms: {700/avg:.2f}x")
print(f"Speedup: {700/avg:.2f}x")
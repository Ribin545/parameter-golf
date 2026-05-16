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

x = torch.randint(0, 1024, (150, 1024), device=device)
y = torch.randint(0, 1024, (150, 1024), device=device)

# Test uncompiled
for _ in range(3):
    model(x, y).backward()
    model.zero_grad(set_to_none=True)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    model(x, y).backward()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
uncompiled = (time.perf_counter()-t0)*1000/5

# Compile max-autotune without cudagraphs
cm = torch.compile(model, mode="max-autotune")
for _ in range(5):
    cm(x, y).backward()
    model.zero_grad(set_to_none=True)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    cm(x, y).backward()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
compiled = (time.perf_counter()-t0)*1000/5

print(f"Uncompiled:   {uncompiled:.1f}ms")
print(f"Max-autotune: {compiled:.1f}ms")
print(f"Speedup:      {uncompiled/compiled:.2f}x")
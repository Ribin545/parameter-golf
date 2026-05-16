import os, time, torch

os.environ.update({
    "MLP_MEMORY_MODE": "off",
    "ATTN_MEMORY_MODE": "off",
    "SDPA_BACKEND": "auto",
})

from model_multilayer import GPTMultiLayer
import torch.nn as nn

device = torch.device("cuda")
model = GPTMultiLayer(
    1024, 5, 512, 8, 4, 2, False, 0.005, 2, 30.0, 10000.0, 3.0,
    4096, 0.05, 8, 2, True, 0.005, 0.08, 0.0
).to(device).bfloat16()

x = torch.randint(0, 1024, (150, 1024), device=device)
y = torch.randint(0, 1024, (150, 1024), device=device)

# Build fake grads
loss = model(x, y)
loss.backward()

# Time just the optimizer step (simulating AdamW)
params = list(model.parameters())
grads = [p.grad for p in params if p.grad is not None]

torch.cuda.synchronize()
t0 = time.perf_counter()
# Simulate clip + step
torch.nn.utils.clip_grad_norm_(params, 1.0)
for p in params:
    if p.grad is not None:
        p.data.add_(p.grad, alpha=-0.01)
torch.cuda.synchronize()
optim_ms = (time.perf_counter()-t0)*1000

print(f"Optimizer step: {optim_ms:.1f}ms")

# Time data loading simulation (just tensor creation)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    _x = torch.randint(0, 1024, (150, 1024), device=device)
    _y = torch.randint(0, 1024, (150, 1024), device=device)
torch.cuda.synchronize()
data_ms = (time.perf_counter()-t0)*1000/10

print(f"Data loading: {data_ms:.1f}ms")

# Time forward+backward
model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
loss = model(x, y)
loss.backward()
torch.cuda.synchronize()
fwd_bwd_ms = (time.perf_counter()-t0)*1000

print(f"Forward+Backward: {fwd_bwd_ms:.1f}ms")
print(f"Total estimated: {fwd_bwd_ms+optim_ms+data_ms:.1f}ms")
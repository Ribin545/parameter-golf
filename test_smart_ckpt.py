import os, time, torch

# Smart config: ATTN ckpt ON (native, no custom autograd), MLP ckpt OFF (Triton handles it)
os.environ.update({
    "MLP_MEMORY_MODE": "off",  # Triton fused MLP has custom autograd — no need for checkpoint
    "ATTN_MEMORY_MODE": "checkpoint",  # Native SDPA benefits from checkpoint
    "SDPA_BACKEND": "auto",
    "MULTILAYER_ACTIVATION_CHECKPOINT": "1",
    "MULTILAYER_ACTIVATION_CHECKPOINT_MODE": "encoder",
    "MLP_RECOMPUTE": "0",  # Triton already handles recompute
})

from model_multilayer import GPTMultiLayer

device = torch.device("cuda")
model = GPTMultiLayer(
    1024, 5, 512, 8, 4, 2, False, 0.005, 2, 30.0, 10000.0, 3.0,
    4096, 0.05, 8, 2, True, 0.005, 0.08, 0.0
).to(device).bfloat16()

x = torch.randint(0, 1024, (150, 1024), device=device)
y = torch.randint(0, 1024, (150, 1024), device=device)

# Warmup
for _ in range(3):
    model(x, y).backward()
    model.zero_grad(set_to_none=True)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    loss = model(x, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
smart_ms = (time.perf_counter()-t0)*1000/5

print(f"Smart ckpt (ATTN only): {smart_ms:.1f}ms")

# Now test baseline
import importlib
import model_multilayer
importlib.reload(model_multilayer)

os.environ.update({
    "MLP_MEMORY_MODE": "checkpoint",
    "ATTN_MEMORY_MODE": "checkpoint",
    "MLP_RECOMPUTE": "1",
})

importlib.reload(model_multilayer)
from model_multilayer import GPTMultiLayer as GPTMultiLayer2

model2 = GPTMultiLayer2(
    1024, 5, 512, 8, 4, 2, False, 0.005, 2, 30.0, 10000.0, 3.0,
    4096, 0.05, 8, 2, True, 0.005, 0.08, 0.0
).to(device).bfloat16()

for _ in range(3):
    model2(x, y).backward()
    model2.zero_grad(set_to_none=True)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    loss = model2(x, y)
    loss.backward()
    model2.zero_grad(set_to_none=True)
torch.cuda.synchronize()
baseline_ms = (time.perf_counter()-t0)*1000/5

print(f"Baseline (all ckpt):    {baseline_ms:.1f}ms")
print(f"Speedup:                {baseline_ms/smart_ms:.2f}x")
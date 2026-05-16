import os, time, torch

os.environ.update({
    "MLP_MEMORY_MODE": "off",
    "ATTN_MEMORY_MODE": "off", 
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

# Warmup
for _ in range(3):
    model(x, y).backward()
    model.zero_grad(set_to_none=True)

# Test WITH Triton
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    loss = model(x, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
with_triton_ms = (time.perf_counter()-t0)*1000/5

# Disable Triton
import model_multilayer
model_multilayer.triton_fused_relu2 = None

# Re-warmup
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
without_triton_ms = (time.perf_counter()-t0)*1000/5

print(f"With Triton:    {with_triton_ms:.1f}ms")
print(f"Without Triton: {without_triton_ms:.1f}ms")
print(f"Delta: {without_triton_ms-with_triton_ms:+.1f}ms")
import os, time, torch

os.environ.update({
    "MLP_MEMORY_MODE": "off",
    "ATTN_MEMORY_MODE": "off",
    "SDPA_BACKEND": "auto",
    "MULTILAYER_ACTIVATION_CHECKPOINT": "0",
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
    loss = model(x, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
uncompiled_ms = (time.perf_counter()-t0)*1000/5

# Compile
compiled_model = torch.compile(model, mode="reduce-overhead")

# Warmup compiled
for _ in range(5):
    compiled_model(x, y).backward()
    model.zero_grad(set_to_none=True)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    loss = compiled_model(x, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
compiled_ms = (time.perf_counter()-t0)*1000/5

print(f"Uncompiled: {uncompiled_ms:.1f}ms")
print(f"Compiled:   {compiled_ms:.1f}ms")
print(f"Speedup:    {uncompiled_ms/compiled_ms:.2f}x")
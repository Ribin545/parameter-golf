import os, time, torch

# Minimal checkpointing: only ATTN (native), no MLP (Triton), no multilayer
os.environ.update({
    "MLP_MEMORY_MODE": "off",
    "ATTN_MEMORY_MODE": "off",  # Even ATTN off — Flash Attention is efficient
    "SDPA_BACKEND": "auto",
    "MULTILAYER_ACTIVATION_CHECKPOINT": "0",
    "MLP_RECOMPUTE": "0",
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
no_ckpt = (time.perf_counter()-t0)*1000/5

# Now with compile
cm = torch.compile(model, mode="reduce-overhead")
for _ in range(5):
    cm(x, y).backward()
    model.zero_grad(set_to_none=True)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    loss = cm(x, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
compiled = (time.perf_counter()-t0)*1000/5

print(f"No ckpt + uncompiled: {no_ckpt:.1f}ms")
print(f"No ckpt + compiled:   {compiled:.1f}ms")
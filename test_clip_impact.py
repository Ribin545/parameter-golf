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

# Warmup
for _ in range(3):
    model.zero_grad(set_to_none=True)
    model(x, y).backward()

# With clip_grad_norm
times_clip = []
for _ in range(5):
    model.zero_grad(set_to_none=True)
    loss = model(x, y)
    loss.backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
    torch.cuda.synchronize()
    times_clip.append((time.perf_counter()-t0)*1000)

# Without clip (just manual zero-grad to simulate)
times_noclip = []
for _ in range(5):
    model.zero_grad(set_to_none=True)
    loss = model(x, y)
    loss.backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Simulate no-op instead of clip
    torch.cuda.synchronize()
    times_noclip.append((time.perf_counter()-t0)*1000)

print(f"clip_grad_norm: {sum(times_clip)/len(times_clip):.1f}ms")
print(f"no clip: {sum(times_noclip)/len(times_noclip):.1f}ms")
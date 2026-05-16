import os, time, torch

os.environ.update({
    "MLP_MEMORY_MODE": "off",
    "ATTN_MEMORY_MODE": "off",
    "MULTILAYER_ACTIVATION_CHECKPOINT": "0",
    "MLP_RECOMPUTE": "0",
    "SDPA_BACKEND": "auto",
})

from model_multilayer import GPTMultiLayer

device = torch.device("cuda")
model = GPTMultiLayer(
    1024, 5, 512, 8, 4, 2, False, 0.005, 2, 30.0, 10000.0, 3.0,
    4096, 0.05, 8, 2, True, 0.005, 0.08, 0.0
).to(device).bfloat16()

cm = torch.compile(model, mode="reduce-overhead", dynamic=False)

x = torch.randint(0, 1024, (150, 1024), device=device)
y = torch.randint(0, 1024, (150, 1024), device=device)

for _ in range(5):
    cm(x, y).backward()
    model.zero_grad(set_to_none=True)

times = []
torch.cuda.synchronize()
for i in range(20):
    t0 = time.perf_counter()
    loss = cm(x, y)
    loss.backward()
    torch.cuda.synchronize()
    dt = (time.perf_counter()-t0)*1000
    model.zero_grad(set_to_none=True)
    if i >= 5:
        times.append(dt)

avg = sum(times)/len(times)
mn = min(times)
mx = max(times)
print(f"dynamic=False: {avg:.1f}ms (min={mn:.1f}, max={mx:.1f})")
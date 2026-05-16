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

cm = torch.compile(model, mode="reduce-overhead")

x = torch.randint(0, 1024, (150, 1024), device=device)
y = torch.randint(0, 1024, (150, 1024), device=device)

# Extended warmup to let compile stabilize
print("Warmup...")
for i in range(20):
    cm(x, y).backward()
    model.zero_grad(set_to_none=True)
    if i % 5 == 0:
        torch.cuda.synchronize()

print("Measuring...")
times = []
torch.cuda.synchronize()
for i in range(30):
    t0 = time.perf_counter()
    loss = cm(x, y)
    loss.backward()
    torch.cuda.synchronize()
    dt = (time.perf_counter()-t0)*1000
    model.zero_grad(set_to_none=True)
    times.append(dt)
    if i < 5:
        print(f"  step {i}: {dt:.1f}ms")

avg_all = sum(times)/len(times)
avg_last = sum(times[-10:])/10
mn = min(times)
mx = max(times)
print(f"\nAll 30 steps: {avg_all:.1f}ms")
print(f"Last 10 steps: {avg_last:.1f}ms")
print(f"Min: {mn:.1f}ms  Max: {mx:.1f}ms")
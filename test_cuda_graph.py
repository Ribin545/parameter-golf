import os, time, torch

os.environ.update({
    "MLP_MEMORY_MODE": "off",  # Must disable for graphs
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

# Static shapes required for CUDA graphs
x = torch.randint(0, 1024, (150, 1024), device=device)
y = torch.randint(0, 1024, (150, 1024), device=device)

# Warmup
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        model(x, y).backward()
        model.zero_grad(set_to_none=True)
torch.cuda.current_stream().wait_stream(s)

# Warmup again on current stream
for _ in range(3):
    model(x, y).backward()
    model.zero_grad(set_to_none=True)

# Capture graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    loss = model(x, y)
    loss.backward()

# Replay
model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    g.replay()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
graph_ms = (time.perf_counter()-t0)*1000/10

# Baseline without graph
model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    loss = model(x, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
baseline_ms = (time.perf_counter()-t0)*1000/10

print(f"Baseline:   {baseline_ms:.1f}ms")
print(f"CUDA Graph: {graph_ms:.1f}ms")
print(f"Speedup:    {baseline_ms/graph_ms:.2f}x")
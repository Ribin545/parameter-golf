import os, time, torch

os.environ.update({
    "MLP_MEMORY_MODE": "checkpoint",
    "ATTN_MEMORY_MODE": "checkpoint",
    "SDPA_BACKEND": "auto",
    "MULTILAYER_ACTIVATION_CHECKPOINT": "1",
    "MULTILAYER_ACTIVATION_CHECKPOINT_MODE": "encoder",
    "MLP_RECOMPUTE": "1",
})

from model_multilayer import GPTMultiLayer

device = torch.device("cuda")
model = GPTMultiLayer(
    1024, 5, 512, 8, 4, 2, False, 0.005, 2, 30.0, 10000.0, 3.0,
    4096, 0.05, 8, 2, True, 0.005, 0.08, 0.0
).to(device).bfloat16()

x = torch.randint(0, 1024, (150, 1024), device=device)
y = torch.randint(0, 1024, (150, 1024), device=device)

for mode in ["default", "reduce-overhead", "max-autotune"]:
    # Fresh model each time
    m = GPTMultiLayer(
        1024, 5, 512, 8, 4, 2, False, 0.005, 2, 30.0, 10000.0, 3.0,
        4096, 0.05, 8, 2, True, 0.005, 0.08, 0.0
    ).to(device).bfloat16()
    
    # Warmup
    for _ in range(3):
        m(x, y).backward()
        m.zero_grad(set_to_none=True)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        loss = m(x, y)
        loss.backward()
        m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    uncompiled = (time.perf_counter()-t0)*1000/5
    
    # Compile
    cm = torch.compile(m, mode=mode)
    for _ in range(5):
        cm(x, y).backward()
        m.zero_grad(set_to_none=True)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        loss = cm(x, y)
        loss.backward()
        m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    compiled = (time.perf_counter()-t0)*1000/5
    
    print(f"{mode:20s}: uncompiled={uncompiled:.1f}ms compiled={compiled:.1f}ms speedup={uncompiled/compiled:.2f}x")
    
    del m, cm
    torch.cuda.empty_cache()
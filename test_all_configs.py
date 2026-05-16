import os, time, torch

# Base config
base_env = {
    "SDPA_BACKEND": "auto",
    "MULTILAYER_ACTIVATION_CHECKPOINT": "1",
    "MULTILAYER_ACTIVATION_CHECKPOINT_MODE": "encoder",
    "MLP_RECOMPUTE": "1",
}

configs = [
    ("No checkpointing", {"MLP_MEMORY_MODE": "off", "ATTN_MEMORY_MODE": "off"}),
    ("MLP+ATTN ckpt", {"MLP_MEMORY_MODE": "checkpoint", "ATTN_MEMORY_MODE": "checkpoint"}),
    ("All ckpt", {"MLP_MEMORY_MODE": "checkpoint", "ATTN_MEMORY_MODE": "checkpoint", "MULTILAYER_ACTIVATION_CHECKPOINT": "1"}),
]

device = torch.device("cuda")

for name, cfg in configs:
    env = {**base_env, **cfg}
    os.environ.update(env)
    
    # Must reimport to pick up env changes
    import importlib
    import model_multilayer
    importlib.reload(model_multilayer)
    from model_multilayer import GPTMultiLayer
    
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
    ms = (time.perf_counter()-t0)*1000/5
    
    print(f"{name}: {ms:.1f}ms")
    
    del model
    torch.cuda.empty_cache()
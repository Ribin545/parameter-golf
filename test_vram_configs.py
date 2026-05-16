import os, torch

def test_config(name, env_vars):
    os.environ.update(env_vars)
    
    # Must reimport fresh
    import importlib
    import model_multilayer
    importlib.reload(model_multilayer)
    from model_multilayer import GPTMultiLayer
    
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = GPTMultiLayer(
        1024, 5, 512, 8, 4, 2, False, 0.005, 2, 30.0, 10000.0, 3.0,
        4096, 0.05, 8, 2, True, 0.005, 0.08, 0.0
    ).to(device).bfloat16()
    
    x = torch.randint(0, 1024, (150, 1024), device=device)
    y = torch.randint(0, 1024, (150, 1024), device=device)
    
    # Forward + backward
    for _ in range(3):
        model(x, y).backward()
        model.zero_grad(set_to_none=True)
    
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    peak_gb = peak_mb / 1024
    
    print(f"{name:35s}: {peak_gb:.2f} GiB")
    
    del model, x, y
    torch.cuda.empty_cache()
    
    return peak_gb

configs = [
    ("All checkpointing ON", {
        "MLP_MEMORY_MODE": "checkpoint", "ATTN_MEMORY_MODE": "checkpoint",
        "MULTILAYER_ACTIVATION_CHECKPOINT": "1", "MLP_RECOMPUTE": "1",
    }),
    ("Smart: ATTN ckpt only", {
        "MLP_MEMORY_MODE": "off", "ATTN_MEMORY_MODE": "checkpoint",
        "MULTILAYER_ACTIVATION_CHECKPOINT": "1", "MLP_RECOMPUTE": "0",
    }),
    ("Minimal: no ckpt at all", {
        "MLP_MEMORY_MODE": "off", "ATTN_MEMORY_MODE": "off",
        "MULTILAYER_ACTIVATION_CHECKPOINT": "0", "MLP_RECOMPUTE": "0",
    }),
]

results = []
for name, env in configs:
    results.append((name, test_config(name, env)))

print(f"\nTotal GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GiB")
print(f"Safety margin: 22 GiB usable")
for name, gb in results:
    safe = "✅ SAFE" if gb < 22 else "❌ OOM RISK"
    print(f"  {name}: {gb:.2f} GiB {safe}")
"""Test: torch.compile on multilayer forward_logits + depth variants."""
from __future__ import annotations
import os, time, torch
from model_multilayer import GPTMultiLayer

torch.cuda.init()
device = torch.device("cuda")

seq_len = 1024
micro_batch = 64

configs = [
    # (dim, layers, steps, mlp_mult, compile, label)
    (384, 5, 2, 3, False, "5Lx2S_d384_m3"),
    (384, 5, 2, 3, True, "5Lx2S_d384_m3+COMPILE"),
    (384, 5, 1, 3, False, "5Lx1S_d384_m3"),
    (384, 5, 1, 4, False, "5Lx1S_d384_m4"),
    (384, 7, 1, 3, False, "7Lx1S_d384_m3"),
    (512, 5, 2, 3, True, "5Lx2S_d512_m3+COMPILE"),
    (512, 5, 2, 3, True, "5Lx2S_d512_m3+C_NO_CUDA"),
]

for dim, layers, steps, mlp_mult, use_compile, label in configs:
    model = GPTMultiLayer(
        vocab_size=1024, num_layers=layers, model_dim=dim,
        num_heads=4, num_kv_heads=2, mlp_mult=mlp_mult,
        tie_embeddings=True, tied_embed_init_std=0.005,
        num_steps=steps, bigram_hash_size=2048, bigram_hash_scale=0.05,
    ).to(device).bfloat16()
    
    total_p = sum(p.numel() for p in model.parameters())
    raw_mb = total_p * 2 / 1024 / 1024
    est_int8 = raw_mb * 0.24
    
    if use_compile:
        if "NO_CUDA" in label:
            mode = "max-autotune-no-cudagraphs"
        else:
            mode = "reduce-overhead"
        print(f"[{label}] compiling forward_logits (mode={mode})...")
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        model.forward_logits = torch.compile(model.forward_logits, mode=mode)
    
    x = torch.randint(0, 1024, (micro_batch, seq_len), device=device)
    y = torch.randint(0, 1024, (micro_batch, seq_len), device=device)
    
    # Warmup (compile triggers here if enabled)
    warmup_iters = 8 if use_compile else 3
    for i in range(warmup_iters):
        try:
            loss = model(x, y)
            loss.backward()
            model.zero_grad()
        except Exception as e:
            print(f"  {label}: COMPILE FAILED - {str(e)[:120]}")
            break
    else:
        # Measure
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 5
        for _ in range(n):
            loss = model(x, y)
            loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000 / n
        
        eff_depth = layers * steps
        print(f"  {label:<30} depth={eff_depth:>2} | fwd+bwd={dt_ms:.0f}ms/ubatch | params={total_p/1e6:.1f}M int8~{est_int8:.1f}MiB {'✓' if est_int8<16 else '✗'}")
    
    del model
    torch.cuda.empty_cache()

print("\n---")
print("Target: <600ms TOTAL step time (fwd+bwd + optimizer)")
print("With grad_accum=1: need fwd+bwd < 550ms for batch_size=(TARGET_TOKENS/1024)")
print("With grad_accum=2: need fwd+bwd < 250ms PER micro-batch")
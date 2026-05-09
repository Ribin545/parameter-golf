"""Quick test: GPTMultiLayer import + parameter budget."""
import sys
sys.path.insert(0, '.')

try:
    import triton_mlp
    print("triton_mlp: OK")
except ImportError as e:
    print(f"triton_mlp: FAIL - {e}")

try:
    import triton_mlp_fallback
    print("triton_mlp_fallback: OK")
except ImportError as e:
    print(f"triton_mlp_fallback: FAIL - {e}")

# Try importing the model
try:
    from model_multilayer import GPTMultiLayer
    print("\nGPTMultiLayer import: OK")
except Exception as e:
    print(f"\nGPTMultiLayer import: FAIL - {e}")
    sys.exit(1)

def test_config(name, dim, layers, mlp_mult, heads=4, kv_heads=2):
    """Test a specific model configuration."""
    model = GPTMultiLayer(
        vocab_size=1024, num_layers=layers, model_dim=dim,
        num_heads=heads, num_kv_heads=kv_heads,
        mlp_mult=mlp_mult, tie_embeddings=True, tied_embed_init_std=0.005,
        bigram_hash_size=3072, ln_scale=True
    )
    total = sum(p.numel() for p in model.parameters())
    
    shared_params = 0
    block_params = 0
    embed_params = 0
    for n, p in model.named_parameters():
        if 'shared' in n:
            shared_params += p.numel()
        elif 'blocks' in n:
            block_params += p.numel()
        else:
            embed_params += p.numel()
    
    print(f"\n--- {name}: dim={dim}, layers={layers}, mlp_mult={mlp_mult} ---")
    print(f"Total: {total:,} params ({total*4/1024/1024:.2f} MiB fp32)")
    print(f"Shared weights: {shared_params:,} ({100*shared_params/total:.1f}%)")
    print(f"Block-specific: {block_params:,} ({100*block_params/total:.1f}%)")
    print(f"Embeddings: {embed_params:,} ({100*embed_params/total:.1f}%)")
    print(f"Compressed (8-bit): {total/1024/1024:.2f} MiB {'✓' if total < 4*1024*1024 else '✗ >4MiB'}")
    
    # Quick GPU forward pass test
    import torch
    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randint(0, 1024, (1, 64), device='cuda')
        y = torch.randint(0, 1024, (1, 64), device='cuda')
        try:
            with torch.no_grad():
                out = model(x, y)
            print(f"GPU forward: OK (loss={out.item():.4f})")
        except Exception as e:
            print(f"GPU forward: FAIL - {e}")

# Test configurations
test_config("Config A", dim=512, layers=11, mlp_mult=4)
test_config("Config B", dim=640, layers=8, mlp_mult=3, heads=8, kv_heads=4)
test_config("Config C", dim=384, layers=16, mlp_mult=4)

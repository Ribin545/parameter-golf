"""Quick GPU test: SharedBlock weight-tied architecture."""
from model_multilayer import GPTMultiLayer
import torch

def test_config(name, dim, layers, mlp_mult, heads=4, kv_heads=2):
    m = GPTMultiLayer(1024, layers, dim, heads, kv_heads, mlp_mult,
                      True, 0.005, bigram_hash_size=2048)
    total = sum(p.numel() for p in m.parameters())
    shared = sum(p.numel() for n, p in m.named_parameters() if 'shared' in n)
    blocks = sum(p.numel() for n, p in m.named_parameters() if 'blocks' in n)
    emb = total - shared - blocks
    mb8 = total / (1024 * 1024)
    print(f"{name}: {total:,} params ({total*4/1e6:.1f}MB fp32, {mb8:.2f}MB int8)")
    print(f"  shared={shared:,}({100*shared/total:.0f}%)  blocks={blocks:,}({100*blocks/total:.0f}%)  emb={emb:,}({100*emb/total:.0f}%)")
    return m

# Config A: dim=512, 11 layers
print("=== Config A ===")
model = test_config("A", 512, 11, 4).cuda()
x = torch.randint(0, 1024, (1, 128), device='cuda')
y = torch.randint(0, 1024, (1, 128), device='cuda')
with torch.no_grad():
    loss = model(x, y)
print(f"GPU fwd loss: {loss.item():.4f}")
model.train()
l = model(x, y)
l.backward()
gn = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
print(f"GPU bwd grad_norm: {gn:.4f}")
del model

# Config B: dim=640, 8 layers, wider
print("\n=== Config B ===")
model = test_config("B", 640, 8, 3, heads=8, kv_heads=4).cuda()
x = torch.randint(0, 1024, (1, 128), device='cuda')
y = torch.randint(0, 1024, (1, 128), device='cuda')
with torch.no_grad():
    l = model(x, y)
print(f"GPU fwd loss: {l.item():.4f}")
model.train()
l = model(x, y)
l.backward()
gn = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
print(f"GPU bwd grad_norm: {gn:.4f}")
del model

# Config C: dim=384, 16 layers, deepest
print("\n=== Config C ===")
model = test_config("C", 384, 16, 4).cuda()
x = torch.randint(0, 1024, (1, 128), device='cuda')
y = torch.randint(0, 1024, (1, 128), device='cuda')
with torch.no_grad():
    l = model(x, y)
print(f"GPU fwd loss: {l.item():.4f}")
model.train()
l = model(x, y)
l.backward()
gn = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
print(f"GPU bwd grad_norm: {gn:.4f}")
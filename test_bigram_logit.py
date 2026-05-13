import torch
from model_multilayer import GPTMultiLayer

# Test 1: Default (disabled)
print("=== Test 1: Default (disabled) ===")
m1 = GPTMultiLayer(
    vocab_size=1024, num_layers=5, model_dim=512, num_heads=8,
    num_kv_heads=4, mlp_mult=2, tie_embeddings=False,
    tied_embed_init_std=0.005, num_steps=2
)
x = torch.randint(0, 1024, (2, 8))
logits1 = m1.forward_logits(x)
print(f"logits shape: {logits1.shape}")
print(f"bigram_logit_enabled: {m1.bigram_logit_enabled}")
print(f"has bigram_logit_bias: {hasattr(m1, 'bigram_logit_bias')}")
print(f"OK\n")

# Test 2: Enabled
print("=== Test 2: Enabled ===")
m2 = GPTMultiLayer(
    vocab_size=1024, num_layers=5, model_dim=512, num_heads=8,
    num_kv_heads=4, mlp_mult=2, tie_embeddings=False,
    tied_embed_init_std=0.005, num_steps=2,
    bigram_logit_enabled=True, bigram_logit_scale_init=0.05
)
logits2 = m2.forward_logits(x)
print(f"logits shape: {logits2.shape}")
print(f"bigram_logit_enabled: {m2.bigram_logit_enabled}")
print(f"has bigram_logit_bias: {hasattr(m2, 'bigram_logit_bias')}")
print(f"has bigram_logit_scale: {hasattr(m2, 'bigram_logit_scale')}")
print(f"bigram_logit_scale value: {float(m2.bigram_logit_scale):.4f}")
print(f"OK\n")

# Test 3: Loss forward (training path)
print("=== Test 3: Loss forward ===")
loss = m2(x, x)  # x as both input_ids and target_ids
print(f"loss: {float(loss):.4f}")
print("OK\n")

# Test 4: Parameter count
print("=== Test 4: Parameter count ===")
total1 = sum(p.numel() for p in m1.parameters())
total2 = sum(p.numel() for p in m2.parameters())
diff = total2 - total1
print(f"Without bigram logit: {total1:,} params")
print(f"With bigram logit:    {total2:,} params")
print(f"Added: {diff:,} params ({diff / 1e6:.2f}M)")
print("OK\n")

print("All tests passed!")
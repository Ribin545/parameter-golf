"""Scientific diagnosis: why the 12-step recurrent model cannot converge."""
import os
os.environ['VOCAB_SIZE'] = '1024'
os.environ['MODEL_DIM'] = '384'
os.environ['RECURRENCE_STEPS'] = '12'
os.environ['LORA_RANK'] = '384'
os.environ['NUM_HEADS'] = '4'
os.environ['NUM_KV_HEADS'] = '2'
os.environ['MLP_MULT'] = '3'
os.environ['DISABLE_COMPILE'] = '1'

from model import GPT
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model = GPT(vocab_size=1024, num_steps=12, model_dim=384, num_heads=4,
            num_kv_heads=2, mlp_mult=3, tie_embeddings=True,
            tied_embed_init_std=0.005)
device = torch.device("cpu")  # CPU analysis
model = model.to(device)

# 1. Parameter analysis: shared vs per-step
shared_names = []
per_step_names = []
for n, p in model.named_parameters():
    if any(x in n for x in ['lora_A', 'lora_B', 'step_embeddings',
                             'v_step_bias', 'attn_scale', 'mlp_scale',
                             'level_gain', 'level_down', 'level_up']):
        per_step_names.append(n)
    else:
        shared_names.append(n)

shared_count = sum(model.get_parameter(n).numel() for n in shared_names)
per_step_count = sum(model.get_parameter(n).numel() for n in per_step_names)
total = shared_count + per_step_count

print("=" * 70)
print("HYPOTHESIS 1: Weight-Tied Depth Is Fundamentally Different")
print("=" * 70)
print(f"  Shared params (used 12×): {shared_count/1e6:.2f}M ({100*shared_count/total:.1f}%)")
print(f"  Per-step params (used 1×): {per_step_count/1e6:.2f}M ({100*per_step_count/total:.1f}%)")
print(f"  Effective virtual depth: 12 layers from 1 weight set")
print()
print("  GRADIENT ANALYSIS:")
print("  - Shared weight grad = sum(grad_1 + grad_2 + ... + grad_12)")
print("  - Per-step weight grad = single step's contribution")
print("  - With clip_norm=3.46: shared grad direction is 12× noisy ensemble")
print("  - Result: shared weights converge to 'average' of 12 steps")
print("    rather than optimal parameters for any single step")

# 2. Forward pass numerical instability analysis
print()
print("=" * 70)
print("HYPOTHESIS 2: Residual Accumulation With Shared Weights")
print("=" * 70)
print("  Standard transformer (12 layers, 12 weight sets):")
print("    x = x + f_i(x, W_i)   # fresh weights per layer")
print("    grad = dL/dx * df_i/dW_i   # clean signal")
print()
print("  Recurrent (12 steps, 1 weight set):")
print("    x = x + f(x, W)  # SAME W applied 12×")
print("    grad = dL/dx * df/dW + dL/dx * df/dW + ... (12×)")
print("    = 12 * (dL/dx * df/dW) on average")
print("  => gradient is 12× but direction is the SAME (not 12 independent)")
print("  => effective learning rate is 12× normal → weight explosion")
print("  => but clip_norm=3.46 constrains → gradient CLIPPED to 1/12 effective")

# 3. The real fix: what would work?
print()
print("=" * 70)
print("HYPOTHESIS 3: Required Architectural Change")
print("=" * 70)
print("  Option A: RECURRENCE_STEPS=1 (standard transformer)")
print("    - Same architecture as 5090 baseline")
print("    - model_dim=1024, micro_batch=32768, grad_accum=16")
print("    - ~1000ms/step on 3090 (verified earlier)")
print()
print("  Option B: 12 independent weight sets (not tied)")
print("    - model_dim=1024 with 12 layers")
print("    - 12× parameter count → doesn't fit 16MB constraint")
print()
print("  Option C: Hybrid — shared attention QKV, independent MLP")
print("    - LoRA already provides per-step adaptation")
print("    - But shared backbone still dominates → same issue")

# 4. Prove with a tiny forward pass
print()
print("=" * 70)
print("EMPIRICAL PROOF: Forward Pass Gradient Scale")
print("=" * 70)

# A single batch
model.train()
x = torch.randint(0, 1024, (1, 128))
y = torch.randint(0, 1024, (1, 128))

# Forward
out = model(x, y)
loss = out.item()
print(f"  Loss: {loss:.4f}")

# Now compute gradient norms for shared vs per-step
out.backward()
shared_grad_norm = 0.0
per_step_grad_norm = 0.0
for n, p in model.named_parameters():
    if p.grad is None:
        continue
    gnorm = p.grad.norm().item()
    if n in shared_names:
        shared_grad_norm += gnorm ** 2
    else:
        per_step_grad_norm += gnorm ** 2

shared_grad_norm = shared_grad_norm ** 0.5
per_step_grad_norm = per_step_grad_norm ** 0.5

print(f"  Shared weight grad norm (total): {shared_grad_norm:.6f}")
print(f"  Per-step param grad norm (total): {per_step_grad_norm:.6f}")
print(f"  Ratio (shared/per-step): {shared_grad_norm/per_step_grad_norm:.2f}×")
print(f"  With clip_norm=3.46: shared kept = {min(1.0, 3.46/shared_grad_norm)*100:.1f}%")

if shared_grad_norm/per_step_grad_norm > 5:
    print()
    print("  CONFIRMED: Shared weight gradient is {:.1f}× larger".format(
        shared_grad_norm/per_step_grad_norm))
    print("  → Gradient clipping DESTROYS shared weight signal")
    print("  → No single clip_norm can balance both param groups")
    print("  → Architecture requires PER-GROUP clipping (not implemented)")
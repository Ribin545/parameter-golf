"""Quick 75-step A/B test for 5-block stage-wise architecture.

NOTE:
The previous version of this test used fully random (x, y) pairs.
That makes learning impossible and the loss will converge to ~ln(vocab)=6.93,
which masks real architectural differences.

This test now uses a *learnable* synthetic next-token task:
  y = (x + 1) % vocab
so we can observe optimization behavior and compare recurrence vs non-recurrence.
"""
import os, sys, time
import torch
from model_multilayer import GPTMultiLayer

def test(name, num_steps, micro_batch=65536):
    print(f"\n{'='*60}")
    print(f"CONFIG: {name} | dim=384 | 5 blocks | recurrence={num_steps}")
    print(f"{'='*60}")
    
    model = GPTMultiLayer(
        vocab_size=1024, num_layers=5, model_dim=384,
        num_heads=3, num_kv_heads=1, mlp_mult=3,
        tie_embeddings=True, tied_embed_init_std=0.005,
        num_steps=num_steps,
        bigram_hash_size=2048
    ).cuda().bfloat16()
    
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} ({total*4/1024/1024:.1f} MB fp32)")
    
    # Simple AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.12, betas=(0.9, 0.95))
    
    t0 = time.time()
    losses = []
    
    vocab = 1024
    for step in range(75):
        # Learnable synthetic task: predict (token+1) mod vocab.
        # This avoids the ln(vocab) floor from random labels and makes the A/B meaningful.
        x = torch.randint(0, vocab, (micro_batch // 256, 256), device='cuda')
        y = (x + 1) % vocab
        
        optimizer.zero_grad()
        loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 10 == 0 or step == 74:
            elapsed = time.time() - t0
            avg_dt = elapsed / (step + 1) * 1000
            print(f"  step:{step} loss:{loss.item():.4f} dt_avg:{avg_dt:.1f}ms")
    
    elapsed = time.time() - t0
    avg_step_ms = elapsed / 75 * 1000
    
    print(f"\nFINAL: step:74 loss:{losses[-1]:.4f} avg_step:{avg_step_ms:.1f}ms")
    return losses[-1], avg_step_ms, losses

# Run Config A: 5 blocks × 2 recurrence
loss_a, dt_a, hist_a = test("A: 5x2 (10 passes)", num_steps=2)
# Clean up
torch.cuda.empty_cache()
time.sleep(2)

# Run Config B: 5 blocks × 1 recurrence
loss_b, dt_b, hist_b = test("B: 5x1 (5 passes)", num_steps=1)

# Report
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"Test                       step:74_loss  avg_step_ms  effective_passes")
print(f"{'-'*60}")
print(f"A: 5 blocks × 2 recur      {loss_a:12.4f}  {dt_a:10.1f}ms    10")
print(f"B: 5 blocks × 1 recur      {loss_b:12.4f}  {dt_b:10.1f}ms    5")
print(f"{'-'*60}")
print(f"HISTORICAL BEST (T4A):     ~3.40 at step 200  ~581ms (dim=512, step-1)")
print(f"TARGET:                     < 2.5 at step 200  < 600ms")
"""75-step A/B: no-LoRA vs QKV-LoRA rank=4 — trial_5090 config."""
import time, torch, sys
torch.cuda.init()
torch.set_float32_matmul_precision("high")
from model_multilayer import GPTMultiLayer

VOCAB = 1024
DIM = 384
HEADS = 6
KV = 3
MLP = 2
LAYERS = 5
STEPS = 2
MB = 65536
SEQ = 1024
B = MB // SEQ  # 64

def run_test(name, lora_rank):
    print(f"\n{'='*60}")
    print(f"CONFIG: {name} | dim={DIM} h={HEADS} kv={KV} mlp={MLP} | 5Lx{STEPS}S | lora={lora_rank}")
    print(f"{'='*60}", flush=True)

    model = GPTMultiLayer(
        vocab_size=VOCAB, num_layers=LAYERS, model_dim=DIM,
        num_heads=HEADS, num_kv_heads=KV, mlp_mult=MLP,
        tie_embeddings=True, tied_embed_init_std=0.005,
        num_steps=STEPS, bigram_hash_size=2048, bigram_hash_scale=0.05,
        lora_rank=lora_rank,
    ).cuda().bfloat16()

    total_p = sum(p.numel() for p in model.parameters())
    mb_raw = total_p * 2 / 1024 / 1024
    print(f"Params: {total_p:,} ({mb_raw:.1f}MiB raw)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.12, betas=(0.9, 0.95))

    t0 = time.time()
    losses = []
    times = []

    for step in range(75):
        x = torch.randint(0, VOCAB, (B, SEQ), device="cuda")
        y = (x + 1) % VOCAB  # learnable synthetic task

        optimizer.zero_grad()
        t_start = time.perf_counter()
        loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        losses.append(loss.item())
        times.append((t_end - t_start) * 1000)

        if step % 15 == 0 or step == 74:
            elapsed = time.time() - t0
            avg_dt = sum(times) / len(times)
            print(f"  step:{step:3d} loss:{loss.item():.4f} step_ms:{times[-1]:.0f} avg_ms:{avg_dt:.0f}",
                  flush=True)

    elapsed = time.time() - t0
    avg_step = sum(times) / len(times)
    med_step = sorted(times)[len(times)//2]

    print(f"\nFINAL: loss={losses[-1]:.4f} avg_ms={avg_step:.0f} med_ms={med_step:.0f} total={elapsed:.1f}s",
          flush=True)
    del model, optimizer
    torch.cuda.empty_cache()
    return losses[-1], avg_step, losses, times

# ─── Run sequentially ───
print("="*60)
print("75-STEP A/B: No LoRA vs QKV-LoRA rank=4")
print(f"Config: dim={DIM}, heads={HEADS}, kv={KV}, mlp={MLP}, 5 layers, {STEPS} recur steps")
print(f"Batch: {B}x{SEQ} = {MB} tokens")
print("="*60, flush=True)

# Test A: no LoRA
loss_a, dt_a, hist_a, t_a = run_test("A: NO-LORA (lora=0)", lora_rank=0)
time.sleep(3)

# Test B: QKV LoRA rank=4
loss_b, dt_b, hist_b, t_b = run_test("B: QKV-LORA rank=4", lora_rank=4)

# ─── Summary ───
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"{'Test':<35} {'step74_loss':>12} {'avg_ms':>8} {'med_ms':>8}")
print(f"{'-'*65}")
print(f"{'A: NO-LORA (lora=0)':<35} {loss_a:12.4f} {dt_a:8.0f} {dt_a:8.0f}")
print(f"{'B: QKV-LORA rank=4':<35} {loss_b:12.4f} {dt_b:8.0f} {dt_b:8.0f}")
print(f"{'-'*65}")
print(f"Target: <600ms avg step | <2.5 loss at step200")

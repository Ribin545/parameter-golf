import os
import time
import torch
from torch.profiler import profile, ProfilerActivity

# Match trial_5090.sh config
os.environ.update({
    "VOCAB_SIZE": "1024", "MODEL_DIM": "512", "NUM_LAYERS": "5",
    "NUM_HEADS": "8", "NUM_KV_HEADS": "4", "MLP_MULT": "2",
    "RECURRENCE_STEPS": "2", "MULTILAYER_LORA_RANK": "8",
    "BIGRAM_HASH_ENABLED": "1", "BIGRAM_HASH_SIZE": "4096",
    "BIGRAM_HASH_SCALE": "0.05", "SHELL_CENTERING_ENABLED": "1",
    "SHELL_CENTERING_LAM": "0.005", "TIE_EMBEDDINGS": "0",
    "MULTILAYER_ACTIVATION_CHECKPOINT": "1",
    "MULTILAYER_ACTIVATION_CHECKPOINT_MODE": "encoder",
    "MLP_RECOMPUTE": "1", "MLP_MEMORY_MODE": "checkpoint",
    "ATTN_MEMORY_MODE": "checkpoint", "SDPA_BACKEND": "auto",
    "MLP_MEMORY_MODE": "off",  # Profile without checkpointing first
    "ATTN_MEMORY_MODE": "off",
})

from model_multilayer import GPTMultiLayer

device = torch.device("cuda")
B = 150
T = 1024

model = GPTMultiLayer(
    vocab_size=1024, num_layers=5, model_dim=512, num_heads=8,
    num_kv_heads=4, mlp_mult=2, tie_embeddings=False,
    tied_embed_init_std=0.005, num_steps=2, logit_softcap=30.0,
    rope_base=10000.0, qk_gain_init=3.0, bigram_hash_size=4096,
    bigram_hash_scale=0.05, lora_rank=8, recurrent_attn_every=2,
    shell_centering_enabled=True, shell_centering_lam=0.005,
    label_smoothing=0.08, z_loss_lambda=0.0
).to(device).bfloat16()

x = torch.randint(0, 1024, (B, T), device=device)
y = torch.randint(0, 1024, (B, T), device=device)

# Warmup
for _ in range(3):
    loss = model(x, y)
    loss.backward()
    model.zero_grad(set_to_none=True)

torch.cuda.synchronize()

# Profile
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    loss = model(x, y)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
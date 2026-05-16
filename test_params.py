import torch, os
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
    "ATTN_MEMORY_MODE": "checkpoint", "SDPA_BACKEND": "auto"
})
from model_multilayer import GPTMultiLayer
m = GPTMultiLayer(
    vocab_size=1024, num_layers=5, model_dim=512, num_heads=8,
    num_kv_heads=4, mlp_mult=2, tie_embeddings=False,
    tied_embed_init_std=0.005, num_steps=2, logit_softcap=30.0,
    rope_base=10000.0, qk_gain_init=3.0, bigram_hash_size=4096,
    bigram_hash_scale=0.05, lora_rank=8, recurrent_attn_every=2,
    shell_centering_enabled=True, shell_centering_lam=0.005,
    label_smoothing=0.08, z_loss_lambda=0.0
)
total = sum(p.numel() for p in m.parameters())
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"total={total:,} trainable={trainable:,}")
for n, p in m.named_parameters():
    print(f"  {n}: {p.numel():,}")
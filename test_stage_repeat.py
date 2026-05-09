import os
import torch


def test_stage_repeat_forward_smoke_cpu():
    # CPU smoke: small model, short seq, just ensure forward runs and loss is finite.
    os.environ["DISABLE_COMPILE"] = "1"
    from model import GPTStageRepeat

    model = GPTStageRepeat(
        vocab_size=128,
        num_stages=3,
        repeats_per_stage=2,
        model_dim=96,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=4,
        tie_embeddings=True,
        tied_embed_init_std=0.01,
        repeat_lora_rank=8,
        bigram_hash_enabled=False,
        shell_centering_enabled=False,
        dropout_p=0.0,
        label_smoothing=0.0,
    )

    B, T = 2, 16
    x = torch.randint(0, 128, (B, T))
    y = torch.randint(0, 128, (B, T))
    loss = model(x, y)
    assert torch.isfinite(loss).item(), f"loss should be finite, got {loss}"


def test_stage_repeat_param_sharing_structure():
    # Ensure per-stage repetition does NOT create separate blocks per repeat.
    from model import GPTStageRepeat

    model = GPTStageRepeat(
        vocab_size=128,
        num_stages=5,
        repeats_per_stage=2,
        model_dim=64,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=4,
        tie_embeddings=True,
        tied_embed_init_std=0.01,
        repeat_lora_rank=4,
        bigram_hash_enabled=False,
        shell_centering_enabled=False,
        dropout_p=0.0,
        label_smoothing=0.0,
    )

    assert len(model.blocks) == 5
    # Each StageBlock has per-repeat tensors indexed by repeats_per_stage
    b0 = model.blocks[0]
    assert b0.repeat_embedding.shape[0] == 2
    assert b0.attn_gate.shape[0] == 2
    assert b0.mlp_gate.shape[0] == 2
    # LoRA adapters should also be per-repeat (num_steps == repeats_per_stage)
    assert b0.mlp.fc.num_steps == 2
    assert b0.mlp.proj.num_steps == 2
    assert b0.attn.proj.num_steps == 2

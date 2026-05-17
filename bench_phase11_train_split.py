"""Phase 11 train-loop timing split for clean 6-layer config.

Times data loading, forward, backward, LR/optimizer, and optional EMA snapshots
over a controlled number of optimizer steps. This helps distinguish true kernel
hot-path drift from optimizer/data/housekeeping drift.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from data_utils import DistributedTokenLoader
from model_multilayer import GPTMultiLayer
from optimizer_utils import Muon


def sync_t() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def main() -> None:
    os.environ.update({
        "BIGRAM_HASH_ENABLED": "0",
        "SHELL_CENTERING_ENABLED": "0",
        "MINI_DEPTH_STATIC": "1",
        "MINI_DEPTH_REFINE_BLOCKS": "3",
        "ATTN_OUTPUT_MODE": "baseline",
        "SDPA_BACKEND": "flash",
        "MLP_MEMORY_MODE": "off",
        "ATTN_MEMORY_MODE": "off",
        "MULTILAYER_ACTIVATION_CHECKPOINT": "0",
        "FUSED_MLP_ENABLED": os.environ.get("FUSED_MLP_ENABLED", "1"),
    })
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda")
    data_path = os.environ.get("DATA_PATH", "/mnt/e/Projects/Proj/golf/data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    train_loader = DistributedTokenLoader(train_files, 0, 1, device)

    model = GPTMultiLayer(
        vocab_size=1024,
        num_layers=6,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        num_steps=2,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        bigram_hash_size=4096,
        bigram_hash_scale=0.05,
        lora_rank=8,
        recurrent_attn_every=1,
        shell_centering_enabled=False,
        label_smoothing=0.0,
        z_loss_lambda=0.0,
    ).to(device).bfloat16().train()
    model.forward_logits = torch.compile(model.forward_logits, mode="default")

    matrix_params, adam_params = [], []
    for n, p in model.named_parameters():
        if p.ndim == 2 and "tok_emb" not in n and "lora" not in n and "bigram_hash" not in n:
            matrix_params.append(p)
        else:
            adam_params.append(p)
    opt_muon = Muon(matrix_params, lr=0.08, momentum=0.95, backend_steps=5)
    opt_adam = torch.optim.AdamW([{"params": adam_params, "lr": 0.015, "weight_decay": 0.0}], betas=(0.9, 0.95), fused=True)

    grad_accum_steps = 5
    micro_tokens = 102400
    seq_len = 1024
    steps = int(os.environ.get("SPLIT_STEPS", "140"))

    print(f"params={sum(p.numel() for p in model.parameters()):,} matrix={len(matrix_params)} adam={len(adam_params)}")
    print("step,total_ms,data_ms,fw_ms,bw_ms,opt_ms,loss")

    for step in range(steps):
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)
        step_loss = 0.0

        t_step0 = sync_t()
        data_ms = fw_ms = bw_ms = 0.0
        for _ in range(grad_accum_steps):
            t0 = sync_t()
            x, y = train_loader.next_batch(micro_tokens, seq_len)
            t1 = sync_t()
            loss = model(x, y)
            t2 = sync_t()
            loss.backward()
            t3 = sync_t()
            data_ms += (t1 - t0) * 1000.0
            fw_ms += (t2 - t1) * 1000.0
            bw_ms += (t3 - t2) * 1000.0
            step_loss += float(loss.detach()) / grad_accum_steps

        t4 = sync_t()
        try:
            opt_muon.step(grad_clip=1.0)
        except TypeError:
            opt_muon.step()
        torch.nn.utils.clip_grad_norm_(adam_params, max_norm=1.0, error_if_nonfinite=False)
        opt_adam.step()
        t5 = sync_t()
        opt_ms = (t5 - t4) * 1000.0
        total_ms = (t5 - t_step0) * 1000.0

        if step < 10 or step % 10 == 0:
            print(f"{step},{total_ms:.2f},{data_ms:.2f},{fw_ms:.2f},{bw_ms:.2f},{opt_ms:.2f},{step_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
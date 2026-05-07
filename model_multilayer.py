"""
Multi-Layer GPT variant for AB testing (Config B).

11 distinct Transformer blocks with:
- BigramHash(3072) token correction
- Value Embedding (VE) on last 2 layers
- LN scale (1/sqrt(layer_idx+1))
- Parallel residual (attn + MLP computed simultaneously)
- Shared QKV weight tensors across layers (weight-tying to save space)

Based on the #1 leaderboard architecture: ValCalib_GPTQ_XSA_BigramHash3072
but stripped of GPTQ/XSA for clean comparison.
"""

from __future__ import annotations
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from triton_mlp import fused_relu2


class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class BigramHashEmbedding(nn.Module):
    """Learned bigram-pair correction on top of token embeddings."""
    def __init__(self, hash_size: int, model_dim: int, scale: float = 0.05, vocab_size: int = 1024):
        super().__init__()
        self.hash_size = hash_size
        self.scale = scale
        self.vocab_size = vocab_size
        self.table = nn.Embedding(hash_size, model_dim)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.002)

    def forward(self, input_ids: Tensor) -> Tensor:
        prev_ids = F.pad(input_ids[:, :-1], (1, 0), value=0)
        hash_idx = (prev_ids.long() * self.vocab_size + input_ids.long()) % self.hash_size
        return self.table(hash_idx) * self.scale


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        cos = freqs.cos()[None, None, :, :].to(dtype=dtype)
        sin = freqs.sin()[None, None, :, :].to(dtype=dtype)
        return cos, sin


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float,
                 gated: bool = False, value_residual: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.value_residual = value_residual
        self.gated = gated

        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor,
                out_w: Tensor, v_embed: Tensor | None = None,
                v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        bsz, seqlen, dim = x.shape

        # Shared weight projections
        q = F.linear(x, q_w).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = F.linear(x, k_w).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if v_embed is not None and v0 is not None:
            # Value embedding correction: project v0 through small embedding
            v_raw = F.linear(x, v_w) + v_embed.unsqueeze(0)
            raw_v = v_raw.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        else:
            v_raw = F.linear(x, v_w)
            raw_v = None
        v = v_raw.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return F.linear(y, out_w), raw_v


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
        x = fused_relu2(x, up_w.t())
        return F.linear(x, down_w)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        layer_idx: int = 0,
        ln_scale: bool = False,
        gated_attention: bool = False,
        value_residual: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        gated=gated_attention, value_residual=value_residual)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor,
                q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor,
                up_w: Tensor, down_w: Tensor,
                v_embed: Tensor | None = None, v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out, raw_v = self.attn(self.attn_norm(x_in) * self.ln_scale_factor,
                                    q_w, k_w, v_w, out_w, v_embed=v_embed, v0=v0)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * \
            self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        return x_out, raw_v


class GPTMultiLayer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float = 10.0,
        rope_base: float = 10000.0,
        qk_gain_init: float = 1.5,
        bigram_hash_size: int = 3072,
        bigram_hash_scale: float = 0.05,
        ln_scale: bool = True,
        gated_attention: bool = False,
        value_residual: bool = False,
        ve_enabled: bool = False,
        ve_dim: int = 128,
        ve_layers: tuple[int, ...] = (9, 10),
    ):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.ve_enabled = ve_enabled
        self.ve_dim = ve_dim
        self.ve_layers = set(ve_layers)
        kv_dim = num_kv_heads * (model_dim // num_heads)

        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        self.bigram_hash = BigramHashEmbedding(bigram_hash_size, model_dim, bigram_hash_scale, vocab_size)

        # Shared QKV projection weights (weight-tying across layers)
        # Stored as (out_dim, in_dim) — F.linear(x, w) computes x @ w.T
        self.shared_q_w = nn.Parameter(torch.empty(model_dim, model_dim))
        self.shared_k_w = nn.Parameter(torch.empty(kv_dim, model_dim))
        self.shared_v_w = nn.Parameter(torch.empty(kv_dim, model_dim))
        self.shared_out_w = nn.Parameter(torch.empty(model_dim, model_dim))
        self.shared_up_w = nn.Parameter(torch.empty(mlp_mult * model_dim, model_dim))
        self.shared_down_w = nn.Parameter(torch.empty(model_dim, mlp_mult * model_dim))

        # Value embedding table (small learned residual for V projection per layer)
        if ve_enabled:
            self.ve_table = nn.Embedding(1, kv_dim)
            nn.init.normal_(self.ve_table.weight, std=0.002)

        # 11 distinct blocks
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  layer_idx=i, ln_scale=ln_scale,
                  gated_attention=gated_attention, value_residual=value_residual)
            for i in range(num_layers)
        ])

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        nn.init.xavier_uniform_(self.shared_q_w, gain=1.0)
        nn.init.xavier_uniform_(self.shared_k_w, gain=1.0)
        nn.init.xavier_uniform_(self.shared_v_w, gain=1.0)
        nn.init.zeros_(self.shared_out_w)
        nn.init.xavier_uniform_(self.shared_up_w, gain=1.0)
        nn.init.zeros_(self.shared_down_w)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = x + self.bigram_hash(input_ids)
        x0 = x

        q_w = self.shared_q_w.to(dtype=x.dtype)
        k_w = self.shared_k_w.to(dtype=x.dtype)
        v_w = self.shared_v_w.to(dtype=x.dtype)
        out_w = self.shared_out_w.to(dtype=x.dtype)
        up_w = self.shared_up_w.to(dtype=x.dtype)
        down_w = self.shared_down_w.to(dtype=x.dtype)

        ve = self.ve_table(torch.tensor([0], device=x.device)).squeeze(0).to(dtype=x.dtype) \
            if self.ve_enabled else None

        for i, block in enumerate(self.blocks):
            v_embed = ve if (self.ve_enabled and i in self.ve_layers) else None
            x, _ = block(x, x0, q_w, k_w, v_w, out_w, up_w, down_w, v_embed=v_embed, v0=None)

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               target_ids.reshape(-1), reduction="mean",
                               label_smoothing=0.05)
        return loss
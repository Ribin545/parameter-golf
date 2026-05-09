"""
Multi-Layer GPT with U-Net skip connections.
Directly modeled after OpenAI's NaiveBaseline architecture:
- CastedLinear (fp32 storage, bf16 compute)
- relu² MLP (no Triton dependency)
- U-Net skip connections (first half stores, second half consumes in reverse)
- Per-block recurrence for additional depth

PERFORMANCE AUDIT (2026-05-09):
- BUG FIX: step_idx was never propagated through Block.forward → attn forward,
  so per-step LoRA was dead code (always using step_idx=0).
- BUG FIX: Cached bf16 casts for LoRA weights + scale tensors to avoid
  repeated .to(dtype) GPU ops every forward call.
- REMOVED: attn_out (proj) LoRA by default — it's the most expensive LoRA
  matmul (dim×dim×rank) and adds ~15-20% overhead per attention op.
  Controlled via separate lora_qkv_rank / lora_proj_rank params.
- QKV LoRA kept at rank=4 (configurable) as quality fallback; small matmul
  overhead (dim×kv_dim×rank) is only ~2-3% of step time.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ─── CastedLinear: fp32 weights, bf16 compute ───
class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    """Rotary embeddings — compile-safe: no forward mutation, precompute once."""
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        self.register_buffer("_cos_full", cos, persistent=False)
        self.register_buffer("_sin_full", sin, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        return self._cos_full[:, :, :seq_len, :].to(device=device, dtype=dtype), \
               self._sin_full[:, :, :seq_len, :].to(device=device, dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class BigramHashEmbedding(nn.Module):
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


class LoRALinear(nn.Module):
    """Per-step LoRA layer wrapping CastedLinear.  rank=0 disables LoRA entirely.

    LoRA:  y = W₀ @ x  +  (α/r) · A @ B @ x
    where A ∈ R^(D_out × r),  B ∈ R^(r × D_in),  α = 1.0

    OPTIMISED: Pre-caches bf16 copies of LoRA weights indexed by step_idx to
    avoid repeated .to(dtype) GPU casts inside the hot forward path.
    """
    def __init__(self, in_features: int, out_features: int, num_steps: int, rank: int = 8):
        super().__init__()
        self.rank = rank
        self.has_lora = rank > 0
        self.num_steps = num_steps
        self.base = CastedLinear(in_features, out_features, bias=False)
        self._lora_cache: list[tuple[Tensor, Tensor]] | None = None
        if self.has_lora:
            self.lora_A = nn.Parameter(torch.empty(num_steps, out_features, rank, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.empty(num_steps, rank, in_features, dtype=torch.float32))
            self.scaling = 1.0 / float(rank)            # α/r
            for s in range(num_steps):
                nn.init.kaiming_uniform_(self.lora_A[s], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[s])

    def _ensure_cache(self, dtype: torch.dtype) -> None:
        """Build bf16 cache once; rebuild if dtype changes (shouldn't normally)."""
        if self._lora_cache is None or (len(self._lora_cache) > 0 and self._lora_cache[0][0].dtype != dtype):
            self._lora_cache = [
                (self.lora_A[s].to(dtype=dtype), self.lora_B[s].to(dtype=dtype))
                for s in range(self.num_steps)
            ]

    def forward(self, x: Tensor, step_idx: int = 0) -> Tensor:
        y = self.base(x)
        if self.has_lora:
            self._ensure_cache(x.dtype)
            a, b = self._lora_cache[step_idx]
            y = y + (x @ b.t() @ a.t()) * self.scaling
        return y


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with separate LoRA ranks for QKV vs attn_out (proj).

    lora_qkv_rank: applied to Q, K, V projections (small matmul overhead)
    lora_proj_rank: applied to output projection (most expensive, removed by default)
    """
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float, num_steps: int = 1,
                 lora_qkv_rank: int = 0, lora_proj_rank: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.has_lora = (lora_qkv_rank > 0) or (lora_proj_rank > 0)
        self.num_steps = num_steps

        # QKV: each is dim×dim (Q) or dim×kv_dim (K,V); kv_dim << dim for GQA
        self.c_q = LoRALinear(dim, dim, num_steps, lora_qkv_rank)
        self.c_k = LoRALinear(dim, kv_dim, num_steps, lora_qkv_rank)
        self.c_v = LoRALinear(dim, kv_dim, num_steps, lora_qkv_rank)
        # proj: dim×dim — most expensive LoRA, defaults to rank=0 (disabled)
        self.proj = LoRALinear(dim, dim, num_steps, lora_proj_rank)
        self.proj.base._zero_init = True

        # Pre-cast q_gain to bf16 once to avoid .to(dtype) in forward
        self._q_gain_bf16: Tensor | None = None
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def _cached_q_gain(self, dtype: torch.dtype) -> Tensor:
        if self._q_gain_bf16 is None or self._q_gain_bf16.dtype != dtype:
            self._q_gain_bf16 = self.q_gain.to(dtype=dtype)[None, :, None, None]
        return self._q_gain_bf16

    def forward(self, x: Tensor, step_idx: int = 0) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x, step_idx=step_idx).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x, step_idx=step_idx).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x, step_idx=step_idx).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self._cached_q_gain(q.dtype)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y, step_idx=step_idx)


class MLP(nn.Module):
    """relu² MLP — matches OpenAI baseline exactly."""
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    """One transformer block with per-step LoRA on attention projections.

    Caches scale tensors in bf16 to avoid repeated .to(dtype) GPU casts.
    """
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: int, rope_base: float, qk_gain_init: float,
                 num_steps: int = 1, lora_qkv_rank: int = 0, lora_proj_rank: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        num_steps=num_steps,
                                        lora_qkv_rank=lora_qkv_rank,
                                        lora_proj_rank=lora_proj_rank)
        self.mlp = MLP(dim, mlp_mult)
        self.has_lora = (lora_qkv_rank > 0) or (lora_proj_rank > 0)
        self.num_steps = num_steps
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        # Cached bf16 views of scale params (built lazily in _ensure_scale_cache)
        self._scale_cache: tuple[Tensor, Tensor, Tensor] | None = None

    def _ensure_scale_cache(self, dtype: torch.dtype) -> None:
        if self._scale_cache is None or self._scale_cache[0].dtype != dtype:
            mix = self.resid_mix.to(dtype=dtype)
            self._scale_cache = (
                mix[0][None, None, :],
                mix[1][None, None, :],
                self.attn_scale.to(dtype=dtype)[None, None, :],
            )

    def forward(self, x: Tensor, x0: Tensor, step_idx: int = 0) -> Tensor:
        self._ensure_scale_cache(x.dtype)
        mix0, mix1, attn_scale_v = self._scale_cache
        x = mix0 * x + mix1 * x0
        # BUG FIX: propagate step_idx so per-step LoRA actually varies across recurrence steps
        attn_out = self.attn(self.attn_norm(x), step_idx=step_idx)
        x = x + attn_scale_v * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPTMultiLayer(nn.Module):
    """Multi-layer GPT with U-Net skip connections + per-block recurrence."""
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 num_heads: int, num_kv_heads: int, mlp_mult: int,
                 tie_embeddings: bool, tied_embed_init_std: float,
                 num_steps: int = 1,
                 logit_softcap: float = 30.0, rope_base: float = 10000.0,
                 qk_gain_init: float = 1.5, bigram_hash_size: int = 2048,
                 bigram_hash_scale: float = 0.05, lora_rank: int = 0):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.lora_rank = lora_rank

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram_hash = BigramHashEmbedding(bigram_hash_size, model_dim, bigram_hash_scale, vocab_size)

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  num_steps=num_steps, lora_qkv_rank=lora_rank, lora_proj_rank=0)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward_logits(self, input_ids: Tensor, **kwargs) -> Tensor:
        """Return logits in shape [B, T, vocab].

        This matches the interface expected by eval_utils.eval_val(), which applies
        cross-entropy over logits.permute(0, 2, 1).
        """
        x = self.tok_emb(input_ids)
        x = x + self.bigram_hash(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        # Pre-cache skip_weights in bf16 (shared across all recurrence steps)
        _skip_cached: list[Tensor] = []
        for wi in range(self.num_skip_weights):
            _skip_cached.append(self.skip_weights[wi][None, None, :])

        # BUG FIX: propagate step_idx so each recurrence pass uses distinct per-step LoRA
        for step in range(self.num_steps):
            skips: list[Tensor] = []
            # Encoder: store skips
            for i in range(self.num_encoder_layers):
                x = self.blocks[i](x, x0, step_idx=step)
                skips.append(x)

            # Decoder: consume skips in reverse order
            for i in range(self.num_decoder_layers):
                if skips and i < self.num_skip_weights:
                    x = x + _skip_cached[i].to(dtype=x.dtype) * skips.pop()
                x = self.blocks[self.num_encoder_layers + i](x, x0, step_idx=step)

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="mean",
        )
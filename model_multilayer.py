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
import os
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from contextlib import nullcontext

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
except Exception:
    sdpa_kernel = None
    SDPBackend = None

try:
    from triton_mlp import fused_relu2 as triton_fused_relu2
except Exception:
    triton_fused_relu2 = None

_MLP_MEMORY_MODE: str | None = None
_ATTN_MEMORY_MODE: str | None = None
_SDPA_BACKEND_MODE: str | None = None
_ATTN_OUTPUT_MODE: str | None = None


def _get_mlp_memory_mode() -> str:
    global _MLP_MEMORY_MODE
    if _MLP_MEMORY_MODE is None:
        mode = os.environ.get("MLP_MEMORY_MODE", "").strip().lower()
        if not mode:
            # Back-compat: old Tier 3.1 flag now maps to checkpoint mode.
            if os.environ.get("MLP_RECOMPUTE", "0") == "1":
                mode = "checkpoint"
            else:
                mode = "off"
        if mode not in {"off", "checkpoint", "eager_recompute"}:
            mode = "off"
        _MLP_MEMORY_MODE = mode
    return _MLP_MEMORY_MODE


def _get_attn_memory_mode() -> str:
    global _ATTN_MEMORY_MODE
    if _ATTN_MEMORY_MODE is None:
        mode = os.environ.get("ATTN_MEMORY_MODE", "off").strip().lower()
        if mode not in {"off", "checkpoint"}:
            mode = "off"
        _ATTN_MEMORY_MODE = mode
    return _ATTN_MEMORY_MODE


def _get_sdpa_backend_mode() -> str:
    global _SDPA_BACKEND_MODE
    if _SDPA_BACKEND_MODE is None:
        mode = os.environ.get("SDPA_BACKEND", "auto").strip().lower()
        if mode not in {"auto", "flash", "mem_efficient", "math"}:
            mode = "auto"
        _SDPA_BACKEND_MODE = mode
    return _SDPA_BACKEND_MODE


def _get_attn_output_mode() -> str:
    global _ATTN_OUTPUT_MODE
    if _ATTN_OUTPUT_MODE is None:
        mode = os.environ.get("ATTN_OUTPUT_MODE", "baseline").strip().lower()
        if mode not in {"baseline", "einsum_fused"}:
            mode = "baseline"
        _ATTN_OUTPUT_MODE = mode
    return _ATTN_OUTPUT_MODE


def _sdpa_context():
    mode = _get_sdpa_backend_mode()
    if sdpa_kernel is None or SDPBackend is None or mode == "auto":
        return nullcontext()
    if mode == "flash":
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION])
    if mode == "mem_efficient":
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    return sdpa_kernel([SDPBackend.MATH])


class ShellCenteringPenalty(nn.Module):
    """Regularises embeddings to stay outside a shell of radius ~sqrt(d-0.1)
    while keeping the centroid near the origin. Prevents drift/collapse.
    From: https://github.com/openai/parameter-golf/issues/2045
    """
    def __init__(self, d_model: int, lam: float = 0.008):
        super().__init__()
        self.lam = lam
        self.d_model = d_model
        self.penalty: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        norms = x.norm(dim=-1, p=2)
        self.penalty = self.lam * (((2 * self.d_model / ((norms ** 2) + 0.1)) ** 0.5) - 1).clamp(min=0).mean()
        norm_of_mean = x.flatten(0, 1).mean(dim=0).norm(p=2)
        self.penalty = self.penalty + self.lam * ((norm_of_mean ** 2) / (self.d_model ** 0.5))
        return x

# ─── CastedLinear: fp32 weights, bf16 compute ───
class CastedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_cache: Tensor | None = None
        self._bias_cache: Tensor | None = None
        self._weight_cache_dtype: torch.dtype | None = None
        self._bias_cache_dtype: torch.dtype | None = None
        self._weight_cache_version: int = -1
        self._bias_cache_version: int = -1

    def forward(self, x: Tensor) -> Tensor:
        # Fast path: if params already match compute dtype/device, avoid cache logic.
        if self.weight.dtype == x.dtype and self.weight.device == x.device:
            weight = self.weight
        else:
            weight_version = self.weight._version
            if (self._weight_cache is None or self._weight_cache_dtype != x.dtype or
                    self._weight_cache_version != weight_version):
                self._weight_cache = self.weight.to(x.dtype)
                self._weight_cache_dtype = x.dtype
                self._weight_cache_version = weight_version
            weight = self._weight_cache

        bias = None
        if self.bias is not None:
            if self.bias.dtype == x.dtype and self.bias.device == x.device:
                bias = self.bias
            else:
                bias_version = self.bias._version
                if (self._bias_cache is None or self._bias_cache_dtype != x.dtype or
                        self._bias_cache_version != bias_version):
                    self._bias_cache = self.bias.to(x.dtype)
                    self._bias_cache_dtype = x.dtype
                    self._bias_cache_version = bias_version
                bias = self._bias_cache

        return F.linear(x, weight, bias)


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
        self._lora_cache_dtype: torch.dtype | None = None
        self._lora_cache_version: tuple[int, int] | None = None
        if self.has_lora:
            self.lora_A = nn.Parameter(torch.empty(num_steps, out_features, rank, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.empty(num_steps, rank, in_features, dtype=torch.float32))
            self.scaling = 1.0 / float(rank)            # α/r
            for s in range(num_steps):
                nn.init.kaiming_uniform_(self.lora_A[s], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[s])

    def _ensure_cache(self, dtype: torch.dtype) -> None:
        """Build bf16 cache lazily; rebuild on dtype or parameter updates."""
        curr_version = (self.lora_A._version, self.lora_B._version)
        if (
            self._lora_cache is None
            or self._lora_cache_dtype != dtype
            or self._lora_cache_version != curr_version
        ):
            self._lora_cache = [
                (self.lora_A[s].to(dtype=dtype), self.lora_B[s].to(dtype=dtype))
                for s in range(self.num_steps)
            ]
            self._lora_cache_dtype = dtype
            self._lora_cache_version = curr_version

    def forward(self, x: Tensor, step_idx: int = 0) -> Tensor:
        y = self.base(x)
        if self.has_lora:
            self._ensure_cache(x.dtype)
            a, b = self._lora_cache[step_idx]
            y = y + (x @ b.t() @ a.t()) * self.scaling
        return y


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with fused QKV + separate LoRA ranks for QKV vs attn_out.

    Tier 1.1: Q, K, V projections fused into a single LoRALinear(dim, dim+2*kv_dim)
    to eliminate 2 redundant input reads and reduce kernel launches.

    lora_qkv_rank: applied to fused QKV projection (small matmul overhead)
    lora_proj_rank: applied to output projection (most expensive, removed by default)
    """
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float, num_steps: int = 1,
                 lora_qkv_rank: int = 0, lora_proj_rank: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        qkv_out_dim = dim + 2 * self.kv_dim  # Q(dim) + K(kv_dim) + V(kv_dim)
        self.has_lora = (lora_qkv_rank > 0) or (lora_proj_rank > 0)
        self.num_steps = num_steps

        # Tier 1.1: fused QKV projection — single matmul instead of 3
        self.c_qkv = LoRALinear(dim, qkv_out_dim, num_steps, lora_qkv_rank)
        # proj: dim×dim — most expensive LoRA, defaults to rank=0 (disabled)
        self.proj = LoRALinear(dim, dim, num_steps, lora_proj_rank)
        self.proj.base._zero_init = True

        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def _cached_q_gain(self, dtype: torch.dtype) -> Tensor:
        # Tier 2 audit change: q_gain is tiny; avoid persistent GPU cache residency.
        q_gain = self.q_gain if self.q_gain.dtype == dtype else self.q_gain.to(dtype=dtype)
        return q_gain[None, :, None, None]

    def _proj_weight_for(self, y: Tensor) -> Tensor:
        weight = self.proj.base.weight
        if weight.dtype != y.dtype or weight.device != y.device:
            weight = weight.to(device=y.device, dtype=y.dtype)
        return weight

    def _project_attn_output(self, y: Tensor, dim: int, step_idx: int) -> Tensor:
        mode = _get_attn_output_mode()
        # Tier 1.3: layout-preserving einsum path avoids contiguous copy
        # Only used when output-proj LoRA is disabled.
        if mode in {"einsum_fused", "baseline"} and not self.proj.has_lora:
            weight = self._proj_weight_for(y)
            weight_4d = weight.view(dim, self.num_heads, self.head_dim).permute(1, 2, 0).contiguous()
            # y: [B, H, T, Dh] -> out: [B, T, D]
            return torch.einsum("bhtd,hdo->bto", y, weight_4d)
        y = y.transpose(1, 2).contiguous().reshape(y.size(0), y.size(2), dim)
        return self.proj(y, step_idx=step_idx)

    def forward(self, x: Tensor, step_idx: int = 0) -> Tensor:
        bsz, seqlen, dim = x.shape
        # Tier 1.1: fused QKV — single linear projection then split
        qkv = self.c_qkv(x, step_idx=step_idx)  # [B, T, dim + 2*kv_dim]
        dim_q, dim_kv = dim, self.kv_dim
        q = qkv[:, :, :dim_q].reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[:, :, dim_q:dim_q+dim_kv].reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[:, :, dim_q+dim_kv:].reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q_gain = self._cached_q_gain(q.dtype)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * q_gain

        with _sdpa_context():
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               enable_gqa=(self.num_kv_heads != self.num_heads))
        return self._project_attn_output(y, dim=dim, step_idx=step_idx)


class MLP(nn.Module):
    """relu² MLP — matches OpenAI baseline exactly.

    Tier 3.1: When MLP_RECOMPUTE=1, uses a custom autograd Function that
    recomputes fc activations during backward instead of storing them,
    saving ~600 MB VRAM per MLP call (at cost of ~0.3ms extra compute).
    """
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def _forward_impl(self, x: Tensor) -> Tensor:
        if triton_fused_relu2 is not None and x.is_cuda:
            x = triton_fused_relu2(x, self.fc.weight.t())
            return self.proj(x)
        x = torch.relu(self.fc(x))
        return self.proj(x.square())

    def forward(self, x: Tensor) -> Tensor:
        mode = _get_mlp_memory_mode()
        if mode == "checkpoint" and self.training and torch.is_grad_enabled():
            return checkpoint(
                lambda z: self._forward_impl(z),
                x,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        # Research-only fallback: mathematically correct but too slow in practice.
        if mode == "eager_recompute" and x.is_cuda and self.training:
            from triton_mlp_recompute import mlp_recompute
            return mlp_recompute(x, self.fc.weight, self.proj.weight)
        return self._forward_impl(x)


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

    def _forward_attn_branch(self, x: Tensor, step_idx: int) -> Tensor:
        return self.attn(self.attn_norm(x), step_idx=step_idx)

    def forward(self, x: Tensor, x0: Tensor, step_idx: int = 0, attend: bool = True,
                mix0: Tensor | None = None, mix1: Tensor | None = None,
                attn_scale_v: Tensor | None = None, mlp_scale_v: Tensor | None = None) -> Tensor:
        # Tier 1.2: accept pre-cast scale tensors to avoid kernel launches
        if mix0 is None:
            mix = self.resid_mix if self.resid_mix.dtype == x.dtype else self.resid_mix.to(dtype=x.dtype)
            mix0, mix1 = mix[0][None, None, :], mix[1][None, None, :]
        if attn_scale_v is None:
            attn_scale = self.attn_scale if self.attn_scale.dtype == x.dtype else self.attn_scale.to(dtype=x.dtype)
            attn_scale_v = attn_scale[None, None, :]
        if mlp_scale_v is None:
            mlp_scale = self.mlp_scale if self.mlp_scale.dtype == x.dtype else self.mlp_scale.to(dtype=x.dtype)
            mlp_scale_v = mlp_scale[None, None, :]
        x = mix0 * x + mix1 * x0
        # BUG FIX: propagate step_idx so per-step LoRA actually varies across recurrence steps
        # ATTENTION GATE: when attend=False, skip attention entirely — MLP-only refine step
        if attend:
            if _get_attn_memory_mode() == "checkpoint" and self.training and torch.is_grad_enabled():
                attn_out = checkpoint(
                    lambda z: self._forward_attn_branch(z, step_idx=step_idx),
                    x,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                attn_out = self._forward_attn_branch(x, step_idx=step_idx)
            x = x + attn_scale_v * attn_out
        x = x + mlp_scale_v * self.mlp(self.mlp_norm(x))
        return x


class GPTMultiLayer(nn.Module):
    """Multi-layer GPT with U-Net skip connections + per-block recurrence."""
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 num_heads: int, num_kv_heads: int, mlp_mult: int,
                 tie_embeddings: bool, tied_embed_init_std: float,
                 num_steps: int = 1,
                 logit_softcap: float = 30.0, rope_base: float = 10000.0,
                 qk_gain_init: float = 1.5, bigram_hash_size: int = 2048,
                 bigram_hash_scale: float = 0.05, lora_rank: int = 0,
                 recurrent_attn_every: int = 1,
                 shell_centering_enabled: bool = False,
                 shell_centering_lam: float = 0.008,
                 label_smoothing: float = 0.0,
                 z_loss_lambda: float = 0.0,
                 bigram_logit_enabled: bool = False,
                 bigram_logit_scale_init: float = 0.05):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.label_smoothing = float(label_smoothing)
        self.z_loss_lambda = float(z_loss_lambda)
        self.shell_centering_enabled = bool(shell_centering_enabled)
        self.shell_centering = (
            ShellCenteringPenalty(model_dim, lam=shell_centering_lam)
            if self.shell_centering_enabled else None
        )
        self.bigram_logit_enabled = bigram_logit_enabled
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.lora_rank = lora_rank
        self.recurrent_attn_every = max(1, int(recurrent_attn_every))
        self.activation_checkpointing = os.environ.get("MULTILAYER_ACTIVATION_CHECKPOINT", "0") == "1"
        self.activation_checkpoint_mode = os.environ.get("MULTILAYER_ACTIVATION_CHECKPOINT_MODE", "full").strip().lower()

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
        self.lm_bias = nn.Parameter(torch.zeros(vocab_size, dtype=torch.float32))
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        
        # Phase 10: learned bigram logit bias — cheap entropy reducer
        if self.bigram_logit_enabled:
            self.bigram_logit_bias = nn.Parameter(torch.zeros(vocab_size, vocab_size, dtype=torch.float32))
            self.bigram_logit_scale = nn.Parameter(torch.tensor(bigram_logit_scale_init, dtype=torch.float32))
        
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _should_checkpoint_block(self, block_idx: int) -> bool:
        if not self.activation_checkpointing:
            return False
        mode = self.activation_checkpoint_mode
        if mode == "full":
            return True
        if mode == "decoder":
            return block_idx >= self.num_encoder_layers
        if mode == "encoder":
            return block_idx < self.num_encoder_layers
        if mode == "encoder_grouped":
            return False
        if mode == "alternate":
            return (block_idx % 2) == 1
        return True

    def _step_uses_attention(self, step_idx: int) -> bool:
        """Safe skip: attend on step 0, MLP-only refine on later steps.
        
        With recurrent_attn_every=2 and 2 steps: step 0 attends, step 1 is MLP-only.
        The last-step safety guard only kicks in if recurrent_attn_every=1 (default).
        """
        if self.recurrent_attn_every <= 1 or self.num_steps <= 1:
            return True
        return (step_idx % self.recurrent_attn_every) == 0

    def _run_block(self, block: Block, block_idx: int, x: Tensor, x0: Tensor, step_idx: int,
                   attend: bool = True,
                   mix0: Tensor | None = None, mix1: Tensor | None = None,
                   attn_scale_v: Tensor | None = None, mlp_scale_v: Tensor | None = None) -> Tensor:
        if self._should_checkpoint_block(block_idx) and self.training and torch.is_grad_enabled():
            return checkpoint(
                lambda a, b: block(a, b, step_idx=step_idx, attend=attend,
                                   mix0=mix0, mix1=mix1,
                                   attn_scale_v=attn_scale_v, mlp_scale_v=mlp_scale_v),
                x,
                x0,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return block(x, x0, step_idx=step_idx, attend=attend,
                     mix0=mix0, mix1=mix1,
                     attn_scale_v=attn_scale_v, mlp_scale_v=mlp_scale_v)

    def _run_encoder_stage(self, x: Tensor, x0: Tensor, step_idx: int,
                           attend: bool = True,
                           mix0: Tensor | None = None, mix1: Tensor | None = None,
                           attn_scale_v: Tensor | None = None,
                           mlp_scale_v: Tensor | None = None,
                           ) -> tuple[Tensor, list[Tensor]]:
        if (
            self.activation_checkpointing
            and self.activation_checkpoint_mode == "encoder_grouped"
            and self.training
            and torch.is_grad_enabled()
        ):
            def encoder_fn(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:
                out = a
                skips: list[Tensor] = []
                for i in range(self.num_encoder_layers):
                    out = self.blocks[i](out, b, step_idx=step_idx, attend=attend,
                                         mix0=mix0, mix1=mix1,
                                         attn_scale_v=attn_scale_v, mlp_scale_v=mlp_scale_v)
                    skips.append(out)
                return (out, *skips)

            enc_out = checkpoint(
                encoder_fn,
                x,
                x0,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            return enc_out[0], list(enc_out[1:])

        skips: list[Tensor] = []
        out = x
        for i in range(self.num_encoder_layers):
            out = self._run_block(self.blocks[i], i, out, x0, step_idx=step_idx, attend=attend,
                                  mix0=mix0, mix1=mix1,
                                  attn_scale_v=attn_scale_v, mlp_scale_v=mlp_scale_v)
            skips.append(out)
        return out, skips

    def forward_logits(self, input_ids: Tensor, **kwargs) -> Tensor:
        """Return logits in shape [B, T, vocab].

        This matches the interface expected by eval_utils.eval_val(), which applies
        cross-entropy over logits.permute(0, 2, 1).
        """
        x = self.tok_emb(input_ids)
        x = x + self.bigram_hash(input_ids)
        if self.shell_centering is not None:
            x = self.shell_centering(x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        # Pre-cache skip_weights in bf16 (shared across all recurrence steps)
        _skip_cached: list[Tensor] = []
        for wi in range(self.num_skip_weights):
            skip_w = self.skip_weights[wi]
            if skip_w.dtype != x.dtype:
                skip_w = skip_w.to(dtype=x.dtype)
            _skip_cached.append(skip_w[None, None, :])

        # ATTENTION GATE: step 0 = full attn+MLP, step 1 = MLP-only refine
        for step in range(self.num_steps):
            attend = self._step_uses_attention(step)
            # Encoder: store skips
            x, skips = self._run_encoder_stage(x, x0, step_idx=step, attend=attend)

            # Decoder: consume skips in reverse order
            for i in range(self.num_decoder_layers):
                if skips and i < self.num_skip_weights:
                    x = x + _skip_cached[i] * skips.pop()
                block_idx = self.num_encoder_layers + i
                x = self._run_block(self.blocks[block_idx], block_idx, x, x0, step_idx=step, attend=attend)

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits_proj = logits_proj + self.lm_bias.to(dtype=logits_proj.dtype)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        
        # Phase 10: learned bigram logit bias
        if self.bigram_logit_enabled:
            prev_ids = F.pad(input_ids[:, :-1], (1, 0), value=0)
            # Use .long() for indexing, .to(dtype) for arithmetic
            bias = self.bigram_logit_bias[prev_ids.long()] * torch.sigmoid(self.bigram_logit_scale)
            logits = logits + bias.to(dtype=logits.dtype)
        
        return logits

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="mean",
            label_smoothing=self.label_smoothing,
        )
        if self.shell_centering is not None:
            loss = loss + self.shell_centering.penalty.to(loss.dtype)
        if self.z_loss_lambda > 0:
            log_z = torch.logsumexp(logits.reshape(-1, logits.size(-1)).float(), dim=-1)
            z_loss = self.z_loss_lambda * (log_z ** 2).mean()
            loss = loss + z_loss
        return loss

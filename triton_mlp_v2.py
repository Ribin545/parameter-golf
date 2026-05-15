import torch
import triton
import triton.language as tl


# Experimental forward-only Triton kernel for end-to-end MLP:
# out = ((x @ w1) -> leaky_relu(0.1)^2) @ w2
#
# This is intentionally isolated from the training path. It is for forward-only
# benchmarking to measure whether a true fused two-gemm MLP has enough upside to
# justify a full backward-capable implementation.


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_D': 64, 'BLOCK_H': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_D': 64, 'BLOCK_H': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_D': 128, 'BLOCK_H': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'D', 'H', 'K'],
)
@triton.jit
def fused_relu2_proj_fwd_kernel(
    x_ptr, w1_ptr, w2_ptr, out_ptr,
    M, K, H, D,
    stride_xm, stride_xk,
    stride_w1k, stride_w1h,
    stride_w2h, stride_w2d,
    stride_om, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    acc_out = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for h0 in range(0, tl.cdiv(H, BLOCK_H)):
        offs_h = h0 * BLOCK_H + tl.arange(0, BLOCK_H)
        acc_hidden = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

        for k0 in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            w1_ptrs = w1_ptr + offs_k[:, None] * stride_w1k + offs_h[None, :] * stride_w1h
            mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            mask_w1 = (offs_k[:, None] < K) & (offs_h[None, :] < H)
            x = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.bfloat16)
            w1 = tl.load(w1_ptrs, mask=mask_w1, other=0.0).to(tl.bfloat16)
            acc_hidden = tl.dot(x, w1, acc_hidden)

        leaky = tl.where(acc_hidden > 0.0, acc_hidden, 0.1 * acc_hidden)
        act = (leaky * leaky).to(tl.bfloat16)

        w2_ptrs = w2_ptr + offs_h[:, None] * stride_w2h + offs_d[None, :] * stride_w2d
        mask_w2 = (offs_h[:, None] < H) & (offs_d[None, :] < D)
        w2 = tl.load(w2_ptrs, mask=mask_w2, other=0.0).to(tl.bfloat16)
        acc_out = tl.dot(act, w2, acc_out)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    mask_o = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    tl.store(out_ptrs, acc_out.to(tl.bfloat16), mask=mask_o)


@torch.no_grad()
def fused_relu2_proj_forward(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Forward-only experimental fused MLP.

    Args:
        x: [..., K]
        w1: [K, H]
        w2: [H, D]
    Returns:
        [..., D]
    """
    assert x.is_cuda and w1.is_cuda and w2.is_cuda, "Triton MLP v2 requires CUDA tensors"
    orig_shape = x.shape
    if x.dim() > 2:
        x2 = x.reshape(-1, orig_shape[-1])
    else:
        x2 = x
    M, K = x2.shape
    K2, H = w1.shape
    H2, D = w2.shape
    assert K == K2 and H == H2
    out = torch.empty((M, D), device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 32), triton.cdiv(D, 64))
    fused_relu2_proj_fwd_kernel[grid](
        x2, w1, w2, out,
        M, K, H, D,
        x2.stride(0), x2.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        out.stride(0), out.stride(1),
    )
    if len(orig_shape) > 2:
        return out.reshape(*orig_shape[:-1], D)
    return out
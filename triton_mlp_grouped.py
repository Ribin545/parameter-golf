import torch
import triton
import triton.language as tl


# ============================================================================
# Grouped / split MLP fusion prototype
# ============================================================================
# Instead of full end-to-end fusion, split hidden dimension into groups.
# For each group:
#   h_group = relu2(x @ w1_group)
#   out += h_group @ w2_group
# This avoids materializing the full hidden matrix while keeping tiling simpler.
#
# Forward-only for now. If forward wins, backward can follow.


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_H": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_H": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_H": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_H": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    ],
    key=["M", "K", "H"],
)
@triton.jit
def fused_relu2_grouped_fwd_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    out_ptr,
    M,
    K,
    H,
    D,
    stride_xm,
    stride_xk,
    stride_w1k,
    stride_w1h,
    stride_w2h,
    stride_w2d,
    stride_om,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    acc_h = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w1_ptrs = w1_ptr + offs_k[:, None] * stride_w1k + offs_h[None, :] * stride_w1h
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w1 = (offs_k[:, None] < K) & (offs_h[None, :] < H)
        x = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.bfloat16)
        w1 = tl.load(w1_ptrs, mask=mask_w1, other=0.0).to(tl.bfloat16)
        acc_h = tl.dot(x, w1, acc_h)

    # Activation: leaky relu(0.1)^2
    leaky = tl.where(acc_h > 0.0, acc_h, 0.1 * acc_h)
    act = (leaky * leaky).to(tl.bfloat16)

    # Write hidden tile to global memory (still needed for partial accumulation strategy)
    # In a full fused kernel we'd accumulate directly into output tiles.
    # For grouped fusion we materialize one tile at a time, then project it.
    # This kernel is intentionally simple to test whether the grouped idea has merit.
    h_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_h[None, :] * stride_od
    mask_h = (offs_m[:, None] < M) & (offs_h[None, :] < H)
    tl.store(h_ptrs, act, mask=mask_h)


@torch.no_grad()
def fused_relu2_grouped_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_size: int = 256,
) -> torch.Tensor:
    """Grouped partial-fusion MLP forward.

    Splits hidden dimension into groups of `group_size`.
    For each group, computes fused fc+activation, then immediately projects.
    This avoids materializing the full hidden matrix in one go.

    Args:
        x: [..., K]
        w1: [K, H]
        w2: [H, D]
        group_size: chunk size for hidden dimension
    Returns:
        [..., D]
    """
    assert x.is_cuda and w1.is_cuda and w2.is_cuda
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

    num_groups = (H + group_size - 1) // group_size
    for g in range(num_groups):
        h_start = g * group_size
        h_end = min(h_start + group_size, H)
        h_len = h_end - h_start

        w1_g = w1[:, h_start:h_end]
        w2_g = w2[h_start:h_end, :]

        # Hidden tile
        h_tile = torch.empty((M, h_len), device=x.device, dtype=torch.bfloat16)
        grid = (triton.cdiv(M, 64), triton.cdiv(h_len, 64))
        fused_relu2_grouped_fwd_kernel[grid](
            x2,
            w1_g,
            w2_g,
            h_tile,
            M,
            K,
            h_len,
            D,
            x2.stride(0),
            x2.stride(1),
            w1_g.stride(0),
            w1_g.stride(1),
            w2_g.stride(0),
            w2_g.stride(1),
            h_tile.stride(0),
            h_tile.stride(1),
        )

        # Immediate projection of this tile
        out += h_tile @ w2_g

    if len(orig_shape) > 2:
        return out.reshape(*orig_shape[:-1], D)
    return out
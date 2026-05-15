import time
import torch

from triton_mlp import fused_relu2


def bench(label, fn, iters=30):
    for _ in range(8):
        out = fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000 / iters
    return label, dt, out


def main():
    device = torch.device("cuda")
    B, T, K, H, D = 150, 1024, 512, 1024, 512
    x = torch.randn(B, T, K, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(K, H, device=device, dtype=torch.bfloat16)
    w2 = torch.randn(H, D, device=device, dtype=torch.bfloat16)

    hidden = fused_relu2(x, w1)
    hidden_flat = hidden.reshape(-1, H)
    x_flat = x.reshape(-1, K)

    cases = []
    cases.append(bench(
        "baseline_linear_3d",
        lambda: torch.nn.functional.linear(fused_relu2(x, w1), w2.t()),
    ))
    cases.append(bench(
        "linear_on_precomputed_hidden_3d",
        lambda: torch.nn.functional.linear(hidden, w2.t()),
    ))
    cases.append(bench(
        "linear_on_flat_hidden_then_reshape",
        lambda: torch.nn.functional.linear(hidden_flat, w2.t()).reshape(B, T, D),
    ))
    cases.append(bench(
        "matmul_on_flat_hidden_then_reshape",
        lambda: (hidden_flat @ w2).reshape(B, T, D),
    ))
    cases.append(bench(
        "einsum_hidden_proj",
        lambda: torch.einsum("bth,hd->btd", hidden, w2),
    ))
    cases.append(bench(
        "fully_flat_pipeline_then_reshape",
        lambda: torch.nn.functional.linear(fused_relu2(x_flat, w1), w2.t()).reshape(B, T, D),
    ))

    ref = cases[0][2]
    print("MLP handoff benchmark")
    print("-" * 60)
    base = cases[0][1]
    for label, dt, out in cases:
        max_diff = (ref - out).abs().max().item()
        print(f"{label:36s} {dt:8.3f} ms   rel={base/dt:6.3f}x   max_diff={max_diff}")


if __name__ == "__main__":
    main()
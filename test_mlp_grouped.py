import time
import torch

from triton_mlp import fused_relu2
from triton_mlp_grouped import fused_relu2_grouped_forward


def bench(label, fn, iters=20):
    for _ in range(6):
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

    cases = []
    cases.append(bench(
        "baseline_split",
        lambda: torch.nn.functional.linear(fused_relu2(x, w1), w2.t()),
    ))

    for gs in [128, 256, 512]:
        cases.append(bench(
            f"grouped_{gs}",
            lambda gs=gs: fused_relu2_grouped_forward(x, w1, w2, group_size=gs),
        ))

    ref = cases[0][2]
    print("Grouped MLP fusion benchmark")
    print("-" * 70)
    base = cases[0][1]
    for label, dt, out in cases:
        max_diff = (ref - out).abs().max().item()
        print(f"{label:20s} {dt:8.3f} ms   rel={base/dt:6.3f}x   max_diff={max_diff}")


if __name__ == "__main__":
    main()
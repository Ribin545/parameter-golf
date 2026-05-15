import time
import torch

from triton_mlp import fused_relu2
from triton_mlp_v2 import fused_relu2_proj_forward


def main():
    device = torch.device("cuda")
    B, T, K, H, D = 150, 1024, 512, 1024, 512
    x = torch.randn(B, T, K, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(K, H, device=device, dtype=torch.bfloat16)
    w2 = torch.randn(H, D, device=device, dtype=torch.bfloat16)

    for _ in range(5):
        y = fused_relu2(x, w1)
        z = torch.nn.functional.linear(y, w2.t())
        _ = fused_relu2_proj_forward(x, w1, w2)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(20):
        y = fused_relu2(x, w1)
        z = torch.nn.functional.linear(y, w2.t())
    torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - t0) * 1000 / 20

    t0 = time.perf_counter()
    for _ in range(20):
        z2 = fused_relu2_proj_forward(x, w1, w2)
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) * 1000 / 20

    ref = torch.nn.functional.linear(fused_relu2(x, w1), w2.t())
    max_diff = (ref - z2).abs().max().item()
    print(f"baseline_forward: {baseline_ms:.2f}ms")
    print(f"fused_v2_forward: {fused_ms:.2f}ms")
    print(f"speedup: {baseline_ms / fused_ms:.3f}x")
    print(f"max_abs_diff: {max_diff}")


if __name__ == "__main__":
    main()
from model_multilayer import GPTMultiLayer
import torch

configs = [
    (5, 384, 2),
    (5, 384, 3),
    (5, 512, 2),
    (7, 384, 2),
    (7, 512, 2),
    (9, 384, 2),
    (11, 384, 1),
    (11, 384, 2),
]
print(f"{'Config':<20} {'Params':>12} {'Raw MiB':>10} {'Est int8 MiB':>14}")
print("-" * 56)
for layers, dim, steps in configs:
    m = GPTMultiLayer(1024, layers, dim, 4, 2, 8, True, 0.005, steps)
    total = sum(p.numel() for p in m.parameters())
    raw_mb = total * 2 / 1024 / 1024
    est_int8 = raw_mb * 0.4
    label = f"{layers}L x{steps}S d={dim}"
    print(f"{label:<20} {total:>12,} {raw_mb:>10.2f} {est_int8:>14.2f}")
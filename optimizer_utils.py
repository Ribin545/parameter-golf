import torch
from torch import Tensor
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 3, eps: float = 1e-7) -> Tensor:
    """
    Polar Express (Quintic Minimax) - Degree 5
    Optimized for GPU throughput and numerical stability.
    """
    dtype = G.dtype
    X = G.float()
    X /= X.norm().add(eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    # Polar Express minimax coefficients (Quintic Degree 5)
    # Optimized for fast convergence and stability in bfloat16.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T # [out_channels, out_channels]
        AX = A @ X
        AAX = A @ AX
        X = a * X + b * AX + c * AAX
        
    X = X.T if transposed else X
    return X.to(dtype)


class Lion(torch.optim.Optimizer):
    """Lion optimizer — sign-based updates with momentum. Fast convergence for small models."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                # Lion update: m = beta1*m + (1-beta1)*g; update = lr * sign(beta2*m + (1-beta2)*g)
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                update = exp_avg.mul(beta2).add(g, alpha=1 - beta2)
                p.add_(update.sign(), alpha=-lr)
                if wd > 0:
                    p.add_(p, alpha=-lr * wd)
        return loss


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )
        # Persistent flat buffer cache per param-group to avoid reallocating a large
        # temporary tensor every optimizer step. Reuse reduces allocator churn and
        # may help long-run dt creep from fragmentation.
        self._flat_buffer_cache: dict[int, tuple[torch.Tensor, int]] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            cache_key = id(group)
            cached = self._flat_buffer_cache.get(cache_key)
            if (
                cached is None
                or cached[1] != total_params
                or cached[0].device != params[0].device
            ):
                updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
                self._flat_buffer_cache[cache_key] = (updates_flat, total_params)
            else:
                updates_flat = cached[0]
                updates_flat.zero_()

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss

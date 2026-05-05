#!/usr/bin/env python
"""eap_fit.py — Exponential Asymptote Projection for A/B test ranking.

Reads a training log file with VAL_LOSS_EVERY output, extracts BPB trajectory,
fits exponential decay model f(t) = L + A*exp(-t/tau), and reports the fitted
asymptote L with confidence intervals.

Pure numpy implementation — no scipy dependency.

Usage: python eap_fit.py <log_path>
"""

import re
import sys
import math
import numpy as np


def _detect_encoding(log_path):
    """Auto-detect file encoding from BOM or fallback to utf-8."""
    with open(log_path, 'rb') as f:
        head = f.read(4)
    if head.startswith(b'\xff\xfe'):
        return 'utf-16-le'
    elif head.startswith(b'\xfe\xff'):
        return 'utf-16-be'
    return 'utf-8'


def parse_bpb_trajectory(log_path):
    """Extract BPB validation points and their step numbers from a log file.

    Matches lines like:
      step:100 val_loss:4.2240 val_bpb:2.4937 ...
      [best] new_best step=100 val_loss=4.2240 val_bpb=2.4937 ...
    Auto-detects UTF-8, UTF-16-LE, UTF-16-BE.
    """
    encoding = _detect_encoding(log_path)
    steps, bpb = [], []
    seen = set()
    with open(log_path, 'r', encoding=encoding, errors='replace') as f:
        for line in f:
            # Try multiple patterns (step: or step= format)
            for pat in [
                r'step:(\d+) .*?val_bpb:([\d.]+)',
                r'step=(\d+) .*?(?:best_)?val_bpb:([\d.]+)',
            ]:
                m = re.search(pat, line)
                if m:
                    s = int(m.group(1))
                    v = float(m.group(2))
                    if s not in seen:
                        steps.append(s)
                        bpb.append(v)
                        seen.add(s)
                    break
    if not steps:
        return np.array([]), np.array([])
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(bpb)[order]


def exp_decay(t, L, A, tau):
    """Exponential decay: L + A * exp(-t / tau)."""
    return L + A * np.exp(-t / tau)


def _fit_loglinear(steps, bpb, L_try):
    """For a fixed L, fit log(A*exp(-t/tau)) = log(A) - t/tau via linear regression.

    Returns (A_fitted, tau_fitted, ss_res) or (None, None, inf) if invalid.
    """
    shifted = bpb - L_try
    if np.any(shifted <= 0):
        return None, None, float('inf')

    y = np.log(shifted)
    # Linear regression: y = c + m * t, where c = log(A), m = -1/tau
    n = len(steps)
    sx = np.sum(steps)
    sy = np.sum(y)
    sxx = np.sum(steps ** 2)
    sxy = np.sum(steps * y)

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None, None, float('inf')

    m = (n * sxy - sx * sy) / denom
    c = (sy - m * sx) / n

    if m >= 0:  # tau would be negative or infinite
        return None, None, float('inf')

    tau = -1.0 / m
    A = math.exp(c)

    # Residual sum of squares
    pred = L_try + A * np.exp(-steps / tau)
    ss_res = np.sum((bpb - pred) ** 2)

    return A, tau, ss_res


def fit_eap(steps, bpb, verbose=True, initial_bpb=None):
    """Fit exponential decay to BPB trajectory via grid search + log-linear least squares.

    No scipy dependency — uses numpy grid search for L, linear regression for A and tau.

    Returns dict with L, L_ci, A, tau, r_sq.
    """
    if initial_bpb is not None:
        steps = np.concatenate([[0], steps])
        bpb = np.concatenate([[initial_bpb], bpb])

    if len(steps) < 5:
        print(f"[WARNING] Only {len(steps)} points - fit may be unreliable")

    bpb_final = bpb[-1]
    bpb_initial = bpb[0]
    A_est = bpb_initial - bpb_final

    # Grid search for L: scan from slightly below final BPB to far below
    # L must be strictly less than all observed BPB values for log-linearization
    L_min = max(0.1, bpb_final - 0.3)
    L_max = min(bpb_final - 0.001, bpb[-1] - 0.001)
    if L_max <= L_min:
        L_max = L_min + 0.001

    n_grid = 200
    L_candidates = np.linspace(L_min, L_max, n_grid)
    best_ss_res = float('inf')
    best_L, best_A, best_tau = None, None, None

    for L_try in L_candidates:
        A, tau, ss_res = _fit_loglinear(steps, bpb, L_try)
        if A is not None and ss_res < best_ss_res:
            best_ss_res = ss_res
            best_L, best_A, best_tau = L_try, A, tau

    if best_L is None:
        print("  [FIT FAILED] No valid L found in grid search")
        return {'L': float('inf'), 'L_ci': float('inf'),
                'A': 0, 'tau': 0, 'r_sq': 0, 'sigma': None,
                'n_points': len(steps)}

    # Refined grid search around best_L
    delta = (L_max - L_min) / n_grid
    refine_min = max(L_min, best_L - 2 * delta)
    refine_max = min(L_max, best_L + 2 * delta)
    L_refined = np.linspace(refine_min, refine_max, 100)
    for L_try in L_refined:
        A, tau, ss_res = _fit_loglinear(steps, bpb, L_try)
        if A is not None and ss_res < best_ss_res:
            best_ss_res = ss_res
            best_L, best_A, best_tau = L_try, A, tau

    L, A, tau = best_L, best_A, best_tau

    # R-squared
    pred = exp_decay(steps, L, A, tau)
    ss_tot = np.sum((bpb - np.mean(bpb)) ** 2)
    r_sq = 1 - best_ss_res / ss_tot if ss_tot > 0 else 0.0

    # Bootstrap confidence interval for L via residual resampling
    residuals = bpb - pred
    n_boot = 500
    L_boot = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        bpb_boot = pred + rng.choice(residuals, size=len(residuals), replace=True)
        # Quick fit for bootstrap sample
        best_ss_boot = float('inf')
        best_L_boot = None
        for L_try in L_candidates[::4]:  # Coarser grid for speed
            A_b, tau_b, ss_b = _fit_loglinear(steps, bpb_boot, L_try)
            if A_b is not None and ss_b < best_ss_boot:
                best_ss_boot = ss_b
                best_L_boot = L_try
        if best_L_boot is not None:
            L_boot.append(best_L_boot)

    sigma_L = float(np.std(L_boot)) if len(L_boot) > 10 else 0.05
    L_ci = 2 * sigma_L

    if verbose:
        print(f"  L  = {L:.4f}  (asymptote - LOWER is better)")
        print(f"  A  = {A:.4f}  (initial deviation above floor)")
        print(f"  tau = {tau:.1f}  (time constant - larger = sustains longer)")
        print(f"  R^2 = {r_sq:.4f}")
        print(f"  sL  = {sigma_L:.4f}  (bootstrap, n={len(L_boot)})")
        print(f"  L 95% CI = [{L - L_ci:.4f}, {L + L_ci:.4f}]")
        for t in [200, 300, 400]:
            p = exp_decay(t, L, A, tau)
            print(f"  Predicted BPB@{t} = {p:.4f}")

    return {
        'L': L, 'L_ci': L_ci,
        'A': A, 'tau': tau,
        'r_sq': r_sq, 'sigma_L': sigma_L,
        'n_points': len(steps)
    }


def compare_candidates(candidates, verbose=True):
    """Compare multiple fitted candidates and return ranking.

    Args:
        candidates: dict of {name: fit_result_dict}
        verbose: print comparison table

    Returns:
        list of (name, L) sorted by L (best first)
    """
    ranking = sorted(candidates.items(), key=lambda x: x[1]['L'])

    if verbose:
        print(f"\n{'=' * 65}")
        print(f"{'Rank':<5} {'Candidate':<20} {'L':>8} {'R²':>8} {'Δ vs best':>12}")
        print(f"{'-' * 65}")
        best_L = ranking[0][1]['L']
        for i, (name, fit) in enumerate(ranking):
            delta = fit['L'] - best_L if fit['L'] != float('inf') else float('inf')
            flag = " ← WINNER" if i == 0 else ""
            print(f"{i + 1:<5} {name:<20} {fit['L']:>8.4f} {fit['r_sq']:>8.4f} "
                  f"{delta:>+10.4f}{flag}")

    return ranking


if __name__ == '__main__':
    log_path = sys.argv[1] if len(sys.argv) > 1 else 'logs/phase5_revert_noShellCentering_out.log'

    print(f"EAP Fit - {log_path}")
    steps, bpb = parse_bpb_trajectory(log_path)
    print(f"  Loaded {len(steps)} validation points")

    if len(steps) == 0:
        print("  [ERROR] No BPB points found in log.")
        print("  Sample of file content:")
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f):
                    if 'val_bpb' in line or 'loss' in line.lower():
                        print(f"    L{i}: {line.rstrip()[:120]}")
                    if i > 500:
                        break
        except Exception as e:
            print(f"  Read error: {e}")
        sys.exit(1)

    print(f"  Range: step {steps[0]}-{steps[-1]}, BPB {bpb[-1]:.4f}-{bpb[0]:.4f}")
    print(f"  Final BPB@{steps[-1]}: {bpb[-1]:.4f}")
    fit_eap(steps, bpb)

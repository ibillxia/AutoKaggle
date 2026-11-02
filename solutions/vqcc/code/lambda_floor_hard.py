"""What is the true hard floor on Lambda implied ONLY by the E_gap gate?

lambda_eff = sqrt(E_gap*(2*Lambda - E_gap)) increases in E_gap, and E_gap must
exceed E_GAP_FLOOR = 3.914898. So for a given Lambda the best lambda_eff sits at
E_gap = floor + tiny. Minimising lambda_eff therefore means minimising Lambda.

The E_gap identity pins  0.5*||W||^2 + 2p > 563.047371  (p = sum of the negative
eigenvalues of h1_prime, i.e. the part of Lambda coming from h1). Hence with
||H2_I|| = ||s|| the H2 part of Lambda is at least 0.25*(sqrt(C - 4p) - ||s||)^2
with C = 1126.0947.  This prints f(p) = p + 0.25*(sqrt(C-4p)-||s||)^2 and its
minimiser, i.e. the LOOSEST possible Lambda floor, to separate 'the floors allow
it' from 'the fit forbids it'.

Usage: python lambda_floor_hard.py [--s 22.8471]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402

C_CONST = 1126.0947  # 2 * 563.047371
S_NORM = 22.8471     # ||sum_b w_b||, measured from the shipped archive


def lam_eff(E_gap: float, Lam: float) -> float:
    return float(np.sqrt(E_gap * (2.0 * Lam - E_gap)))


def main() -> None:
    s = S_NORM
    if '--s' in sys.argv:
        s = float(sys.argv[sys.argv.index('--s') + 1])
    floor = V.E_GAP_FLOOR
    print(f'E_GAP_FLOOR = {floor}, ||s|| = {s:.4f}, C = {C_CONST}')
    print()
    print(f"{'p':>10} {'Lambda floor':>14} {'lam_eff@floor':>14}")
    best = (1e18, None)
    for p in (0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0, 150.0, 200.0, 240.0):
        root = np.sqrt(max(C_CONST - 4.0 * p, 0.0))
        lam = p + 0.25 * (root - s) ** 2
        le = lam_eff(floor + 1e-9, lam) if lam > floor / 2 else float('nan')
        flag = ''
        if lam < best[0]:
            best = (lam, p)
            flag = '  <- min'
        print(f'{p:>10.1f} {lam:>14.4f} {le:>14.4f}{flag}')
    # Analytic minimiser of g(p) = p + 0.25*(sqrt(C-4p) - s)^2 on [0, C/4).
    ps = np.linspace(0.0, C_CONST / 4 - 1e-6, 2_000_001)
    g = ps + 0.25 * (np.sqrt(C_CONST - 4.0 * ps) - s) ** 2
    i = int(np.argmin(g))
    print()
    print(f'grid min: Lambda >= {g[i]:.4f} at p = {ps[i]:.4f}')
    print(f'  -> lambda_eff >= {lam_eff(floor + 1e-9, g[i]):.4f}')
    print(f'  shipped: Lambda = 293.5256, lambda_eff = 47.85037 '
          f'(E_gap 3.926532)')
    cur = lam_eff(floor + 1e-9, 293.5256)
    print(f'  if only E_gap were pushed to its floor: lambda_eff = {cur:.4f} '
          f'(x{47.85037 / cur:.5f})')
    print(f'  composite gain of the whole unreachable floor: '
          f'x{(47.85037 / lam_eff(floor + 1e-9, g[i])) ** (1 / 3):.4f}')


if __name__ == '__main__':
    main()

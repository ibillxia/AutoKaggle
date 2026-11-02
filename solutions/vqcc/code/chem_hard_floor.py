"""The hard floor on Lambda implied by the E_gap inequality alone.

lambda_h2 = 0.25 * sum_rc (L1_rc + |H2_I_rc|)^2 >= 0.25 * ||H2_I||^2,
because both L1_rc and |H2_I_rc| are nonnegative. H2_I = W + s with
s = sum_b w_b, so ||H2_I|| >= ||W|| - ||s||. The E_gap floor gives

    E_gap > 3.914898  <=>  0.5*||W||^2 - 2p > 563.0474

with p = sum|negative eigenvalues of h1_prime| >= 0, hence

    ||W||^2 > 1126.0948 - 4p.

Putting the two together,

    Lambda = lambda_h1 + lambda_h2 >= p + 0.25*(sqrt(1126.0948 - 4p) - ||s||)^2

for every admissible W. This script evaluates that floor and reports how close the
submitted point is to it, which is the honest statement of how much chemistry
headroom remains.

Usage: python chem_hard_floor.py [SUBMISSION_NPZ]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401

sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as M  # noqa: E402

E_UPPER = -22140.27335
GAP_FLOOR = 3.914898
TWO_NEED = 2.0 * (GAP_FLOOR - (E_UPPER - 0.0))  # placeholder, recomputed below


def main() -> None:
    npz = (Path(sys.argv[1]) if len(sys.argv) > 1
           else env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz')
    z = np.load(npz)
    inp = V.load_inputs(npz)

    a_const = inp.ref['ecore'] - inp.ref['electrons'] * inp.beta_1
    # E_gap = E_upper - E_SOS, E_SOS = a_const + 2P - 0.5||W||^2, P = -p
    # => E_gap = E_upper - a_const + 2p + 0.5||W||^2
    need_half = GAP_FLOOR - (E_UPPER - a_const)          # = 563.0474
    print(f'a_const = ecore - eta*beta_1        = {a_const:.6f}')
    print(f'E_gap > {GAP_FLOOR}  <=>  0.5||W||^2 + 2p > {need_half:.6f}')
    print(f'                       <=>  ||W||^2 > {2 * need_half:.4f} - 4p')
    print()

    X = M.reconstruct_x(z['U'], inp.w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    h1 = M.build_h1_prime(inp.ref['h1_thc'], inp.S, inp.beta_1,
                          inp.ref['electrons'], inp.W, X)
    ev = np.linalg.eigvalsh(h1)
    p = -float(ev[ev < 0].sum())
    s = inp.w.sum(axis=1)
    h2_i = inp.W + s
    lam_h1 = float(np.abs(ev).sum())
    lam_h2 = 0.25 * float(np.sum((np.abs(h2_i) + np.abs(inp.w).sum(axis=1)) ** 2))
    lam = lam_h1 + lam_h2

    print('=== submitted point ===')
    print(f'  p = sum|neg eig|                  = {p:.6f}')
    print(f'  ||W||^2                           = {np.sum(inp.W ** 2):.4f}  '
          f'(required > {2 * need_half - 4 * p:.4f})')
    print(f'  ||s|| = ||sum_b w_b||             = {np.linalg.norm(s):.4f}')
    print(f'  ||H2_I||                          = {np.linalg.norm(h2_i):.4f}')
    print(f'  lambda_h2                         = {lam_h2:.4f}  '
          f'(0.25*||H2_I||^2 = {0.25 * np.sum(h2_i ** 2):.4f})')
    print(f'  Lambda                            = {lam:.4f}')
    print()

    # The floor as a function of p (p is bounded above: ||W||^2 > 0).
    print('=== floor over all admissible W, as a function of p ===')
    print(f"{'p':>10} {'||W||_min':>11} {'||H2I||_min':>12} {'Lambda_floor':>13} {'lam_eff':>9}")
    best = None
    p_max = need_half / 2.0  # beyond this, ||W||^2 > 0 is vacuous... no: 1126-4p>0
    for pp in np.linspace(0.0, 2 * need_half / 4.0 - 1e-9, 41):
        w2_min = 2 * need_half - 4 * pp
        if w2_min <= 0:
            continue
        h2_min = max(0.0, math.sqrt(w2_min) - np.linalg.norm(s))
        lf = pp + 0.25 * h2_min ** 2
        le = (math.sqrt(GAP_FLOOR * (2 * lf - GAP_FLOOR))
              if 2 * lf > GAP_FLOOR else float('nan'))
        if best is None or lf < best[0]:
            best = (lf, pp, le)
        if pp * 10 % 20 < 1e-9:
            print(f'{pp:10.2f} {math.sqrt(w2_min):11.4f} {h2_min:12.4f} {lf:13.4f} {le:9.4f}')
    print()
    print(f'  minimum over p: Lambda >= {best[0]:.4f} at p = {best[1]:.3f} '
          f'-> lambda_eff {best[2]:.4f}')
    print(f'  submitted Lambda = {lam:.4f}, lambda_eff = '
          f'{math.sqrt(3.9265323395266023 * (2 * lam - 3.9265323395266023)):.4f}')
    print(f'  => remaining chemistry headroom x'
          f'{math.sqrt(3.9265323395266023 * (2 * lam - 3.9265323395266023)) / best[2]:.4f}')


if __name__ == '__main__':
    main()

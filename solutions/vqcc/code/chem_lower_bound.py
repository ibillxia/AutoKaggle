"""The factorization-independent floor on Lambda, stated as one inequality.

Three facts from the verifier fix the chemistry axis more tightly than the
individual proposals in the SCORECARD suggest:

1. ``lambda_h2 = 0.25 * sum_rc (L1_rc + |H2_I_rc|)^2`` with
   ``H2_I_rc = W_rc + sum_b w_b`` and ``L1_rc = sum_b |w_b|``. Both terms are
   nonnegative, so ``lambda_h2 >= 0.25 * sum_rc L1_rc^2`` regardless of ``W``.

2. ``L1_rc = sum_b |w_b| >= ||X_rc||_*`` (nuclear norm) because
   ``X_rc = sum_b w_b u_b u_b^T`` is a weighted sum of rank-one projectors and
   the nuclear norm is the dual of the spectral norm:  for any unit ``v``,
   ``v^T X_rc v = sum_b w_b (u_b . v)^2 <= max_b |w_b|``, while
   ``sum_b |w_b| >= max_b |w_b|``. Squaring and summing is looser but valid.

3. ``tr(M) = sum_pq M_pqpq`` with ``M = tei + 0.5(S x I) + 0.5(I x S)`` equals
   ``tr(sum_rc X_rc x X_rc) = sum_rc ||X_rc||_F^2`` to within ``error_2``
   accuracy, and ``tr(M)`` is fixed by the FCIDUMP and the verifier's shifts.

So ``0.25 * tr(M)`` is a floor on ``lambda_h2`` that no factorization can beat,
and ``Lambda >= 0.25 * tr(M) + sum|lambda_j|``. The E_gap floor adds
``0.5||W||^2 > 563.0474 - 2p`` where ``p = sum|negative eigenvalues|``, tying the
two halves together.

Usage: python chem_lower_bound.py [SUBMISSION_NPZ]
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
LAM_FLOOR = 49.0


def main() -> None:
    npz = (Path(sys.argv[1]) if len(sys.argv) > 1
           else env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz')
    z = np.load(npz)
    inp = V.load_inputs(npz)
    num_orbitals = 76

    # --- the target tensor M and its trace -----------------------------------
    target = inp.ref['tei'].copy()
    for p in range(num_orbitals):
        target[p, p, :, :] += 0.5 * inp.S
        target[:, :, p, p] += 0.5 * inp.S
    tr_m = float(np.einsum('pqpq->', target))

    # --- the submitted point --------------------------------------------------
    X = M.reconstruct_x(z['U'], inp.w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    h1 = M.build_h1_prime(inp.ref['h1_thc'], inp.S, inp.beta_1,
                          inp.ref['electrons'], inp.W, X)
    ev = np.linalg.eigvalsh(h1)
    p_neg = -float(ev[ev < 0].sum())
    s = inp.w.sum(axis=1)
    l1 = np.abs(inp.w).sum(axis=1)
    lam_h2 = 0.25 * float(np.sum((np.abs(inp.W + s) + l1) ** 2))
    lam = float(np.abs(ev).sum()) + lam_h2

    frob = float(np.sum(X ** 2))
    sv = np.linalg.svd(X.reshape(-1, num_orbitals, num_orbitals), compute_uv=False)
    nuc = sv.sum(axis=1)

    print('=== trace identity (ties the floor to the FCIDUMP alone) ===')
    print(f'  tr(M) = sum_pq M_pqpq             = {tr_m:.6f}')
    print(f'  submitted sum_rc ||X_rc||_F^2     = {frob:.6f}')
    print(f'  difference                        = {tr_m - frob:+.6f} '
          f'(error_2 is {1e3 * 0.019979918024882685:.2f} mHa, so they agree)')
    print()
    print('=== floors on lambda_h2 ===')
    print(f'  W cancels H2_I:        0.25*sum L1^2       = {0.25 * np.sum(l1 ** 2):.4f}')
    print(f'  plus L1 >= nuclear:    0.25*sum nuc^2      = {0.25 * np.sum(nuc ** 2):.4f}')
    print(f'  plus tr identity:      0.25*tr(M)          = {0.25 * tr_m:.4f}')
    print(f'  submitted lambda_h2                        = {lam_h2:.4f}')
    print()
    print('=== the E_gap coupling ===')
    a_const = inp.ref['ecore'] - inp.ref['electrons'] * inp.beta_1
    need = GAP_FLOOR - (E_UPPER - a_const)
    print(f'  E_gap > {GAP_FLOOR}  <=>  0.5*||W||^2 > {need:.6f} - 2p')
    print(f'  submitted: p = {p_neg:.4f}, so 0.5*||W||^2 > {need - 2 * p_neg:.4f}')
    print(f'             actual 0.5*||W||^2 = {0.5 * np.sum(inp.W ** 2):.4f} '
          f'(margin {0.5 * np.sum(inp.W ** 2) - (need - 2 * p_neg):.4f})')
    print()
    print('=== what that leaves ===')
    print(f'  submitted Lambda  = {lam:.4f}   -> lambda_eff {math.sqrt(3.9265323395266023 * (2 * lam - 3.9265323395266023)):.4f}')
    print(f'  absolute Lambda floor (ignoring E_gap) = '
          f'{0.25 * tr_m:.4f} + sum|lambda_j| >= {0.25 * tr_m:.4f}')
    print(f'  lambda_eff at (Lambda={max(LAM_FLOOR, 0.25 * tr_m):.2f}, E_gap={GAP_FLOOR}) = '
          f'{math.sqrt(GAP_FLOOR * (2 * max(LAM_FLOOR, 0.25 * tr_m) - GAP_FLOOR)):.4f}')
    print()
    print('  The E_gap term is what stops Lambda reaching that floor: it forces')
    print(f'  0.5||W||^2 ~ {need:.0f}, and W enters lambda_h2 through |H2_I| = |W + sum w|.')


if __name__ == '__main__':
    main()

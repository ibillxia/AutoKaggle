"""Can W_rc be chosen so that h1_prime is small?

h1_prime = C0 - sum_rc W_rc X_rc, with C0 = h1_thc - (eta/2) S + beta_1 I fixed
and W_rc = H2_I_rc - sum_b w_b a *submitted* array (the verifier freezes beta_1
and S, not H2_I). X_rc is fixed by the two-body rows, so this is a linear
least-squares problem in the 1950 weights.

If h1_prime can be driven to zero then lambda_h1 = 0 and, because
reconstruct_h1_thc inverts the same identity, error_1 = 0 exactly. Lambda is
then only lambda_h2 = 0.25 sum_rc (|H2_I| + L1(w_rc))^2, whose floor at fixed w
is 0.25 sum L1^2.

Usage: python h1p_null.py [SUBMISSION_NPZ]
"""
from __future__ import annotations

import json
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
FLOOR_LAM, FLOOR_GAP = 49.0, 3.914898


def sym_basis(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs and weights that flatten a symmetric matrix isometrically."""
    iu = np.triu_indices(n)
    wgt = np.where(iu[0] == iu[1], 1.0, math.sqrt(2.0))
    return np.stack(iu, axis=1), wgt


def main() -> None:
    npz = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz')
    z = np.load(npz)
    inp = V.load_inputs(npz)
    ref, w = inp.ref, inp.w
    N = int(ref['orbitals'])
    eta = int(ref['electrons'])
    S, beta_1, ecore = inp.S, inp.beta_1, ref['ecore']
    R, B, C = w.shape

    # X from the submitted two-body rows, exactly as the verifier builds it.
    vectors = M.reconstruct_u_vectors(
        np.asarray(z['angle_words'], dtype=np.float64) / 2.0 ** 9, z['tree_edges'])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    _one, u_b_rho = M.split_u_vectors(vectors, w)
    X = M.reconstruct_x(u_b_rho, w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)

    C0 = ref['h1_thc'] - 0.5 * eta * S + beta_1 * np.eye(N)
    idx, wgt = sym_basis(N)
    A = X[:, :, idx[:, 0], idx[:, 1]] * wgt            # (R, C, M)
    A = A.reshape(R * C, -1)                           # (1950, 2926)
    b = C0[idx[:, 0], idx[:, 1]] * wgt
    W_flat, *_ = np.linalg.lstsq(A.T, b, rcond=None)
    resid = A.T @ W_flat - b
    W_opt = W_flat.reshape(R, C)
    h1p = C0 - np.einsum('rc,rcpq->pq', W_opt, X)
    ev = np.linalg.eigvalsh(h1p)
    print(f'N={N} eta={eta} R*C={R*C} symmetric dim={A.shape[1]}')
    print(f'||C0||_F = {np.linalg.norm(C0):.6f}')
    print(f'least-squares residual ||sum W X - C0||_F = {np.linalg.norm(resid):.6f}'
          f'  (relative {np.linalg.norm(resid)/np.linalg.norm(C0):.3e})')
    print(f'h1_prime eigenvalues: min {ev.min():.4f} max {ev.max():.4f} '
          f'sum {ev.sum():.4f} sum|.| {np.abs(ev).sum():.4f}')

    l1 = np.linalg.norm(w, ord=1, axis=1)
    sum_b = w.sum(axis=1)
    H2 = W_opt + sum_b
    lam_h2 = 0.25 * np.sum((np.abs(H2) + l1) ** 2)
    lam_h1 = float(np.abs(ev).sum())
    Lam = lam_h1 + lam_h2
    E_SOS = ecore - eta * beta_1 - (lam_h1 - ev.sum()) - 0.5 * float(np.sum(W_opt ** 2))
    E_gap = E_UPPER - E_SOS
    ok_dom = 0.0 < E_gap <= 2 * Lam
    lam_eff = math.sqrt(E_gap * (2 * Lam - E_gap)) if ok_dom else float('nan')
    print()
    print(f'lambda_h1 {lam_h1:.4f}  lambda_h2 {lam_h2:.4f}  Lambda {Lam:.4f}')
    print(f'E_SOS {E_SOS:.6f}  E_gap {E_gap:.6f}  (floor {FLOOR_GAP})')
    print(f'Lambda floor {FLOOR_LAM}; 0.25*sum L1^2 = {0.25*np.sum(l1**2):.4f}')
    print(f'lambda_eff {lam_eff:.4f}   vs current 47.8504  -> '
          f'{47.85036566452195/lam_eff if lam_eff == lam_eff else float("nan"):.3f}x')
    np.save(env.SCRATCH / 'W_opt.npy', W_opt)


if __name__ == '__main__':
    main()

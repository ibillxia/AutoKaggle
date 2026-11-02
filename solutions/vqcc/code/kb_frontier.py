"""SUPERSEDED -- kept only as a record; do not run as evidence.

This attempted a (K, B) frontier for the spectral factorization but (a) it has a
dead `if False` branch, and (b) more importantly it measures the wrong thing:
rank truncation of the *eigenelements of T* is irrelevant because the spectral
route already fails on the nuclear-norm budget, not on accuracy.

The results that closed this route are in spectral_budget.py (lambda_h2 budget
4.07x shipped) and m_spec_sqrt.py (spectrum). The (K, B) pairwise truncation is
dominated by the cleaner rank_probe.py measurement on the shipped X.

Original docstring follows.

error_2 of the exact spectral factorization, truncated to (K terms, rank B).

T = sum_i c_i phi_i phi_i^T is exact with error_2 = 0. Two truncations then
apply, and they are independent:

  * keep only the K largest |c_i|   -> tail error  ||c_{K:}||
  * represent each X_i = sqrt(c_i) unvec(phi_i) with rank <= B -> SVD truncation

Because the target T is fixed, total error_2 <= sqrt(tail^2 + rankerr^2)
(Cauchy-Schwarz on the cross term is not tight, so this is measured directly).
This prints the (K, B) frontier so the cost model can be applied to real shapes.

Usage: python kb_frontier.py [--maxk 600]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402


def build_T(inp) -> np.ndarray:
    n = inp.num_orbitals
    tei = np.asarray(inp.ref['tei'], dtype=np.float64)
    S = np.asarray(inp.S, dtype=np.float64)
    idx = np.arange(n)
    tt = tei.reshape(n, n, n, n).copy()
    tt[:, :, idx, idx] += 0.5 * S[:, :, None]
    tt[idx, idx, :, :] += 0.5 * S[None, :, :]
    return tt.reshape(n * n, n * n)


def main() -> None:
    npz = env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz'
    inp = V.load_inputs(npz)
    n = inp.num_orbitals
    T = build_T(inp)
    T = 0.5 * (T + T.T)

    # Top K eigenpairs once.
    maxk = 600
    if '--maxk' in sys.argv:
        maxk = int(sys.argv[sys.argv.index('--maxk') + 1])
    ev, evec = np.linalg.eigh(T)
    order = np.argsort(-np.abs(ev))[:maxk]
    c = ev[order]
    Phi = evec[:, order]                      # (n^2, K)
    print(f'N={n}  K_max={maxk}  kept mass {100 * np.abs(c).sum() / np.abs(ev).sum():.4f}%')

    # Reconstruct T from the truncated set once; residual to full T is the tail.
    PhiK = Phi * np.sqrt(np.abs(c))[None, :]   # columns are v_i
    recon_K = PhiK @ PhiK.T
    tail_err = float(np.linalg.norm(T - recon_K))

    print()
    print(f"{'K':>5} {'B':>4} {'tail mHa':>10} {'rank mHa':>10} {'total mHa':>11} {'passes':>7}")
    for K in (200, 300, 400, 480, 600):
        if K > maxk:
            continue
        Vk = PhiK[:, :K]
        for B in (76, 64, 56, 48, 40, 32):
            acc = np.zeros_like(T)
            for i in range(K):
                M = Vk[:, i].reshape(n, n)
                u, s, vt = np.linalg.svd(M)
                Mi = (u[:, :B] * s[:B]) @ vt[:B, :]
                acc += np.outer(Mi.ravel(), Mi.ravel())
            tot = float(np.linalg.norm(T - acc))
            # rank-only error (against the K-term target, not full T)
            rank_err = float(np.linalg.norm(recon_K_from(Vk, K, n) - acc)) if False else np.nan
            print(f'{K:>5} {B:>4} {1e3 * tail_err:>10.4f} {"":>10} '
                  f'{1e3 * tot:>11.4f} {("yes" if tot <= 0.020 else "NO"):>7}', flush=True)


def recon_K_from(Vk, K, n):
    return Vk[:, :K] @ Vk[:, :K].T


if __name__ == '__main__':
    main()

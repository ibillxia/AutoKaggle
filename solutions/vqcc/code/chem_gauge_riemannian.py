"""Full orthogonal gauge descent on the chemistry factor (Riemannian CG).

The verifier's Lambda charges 1/4 sum_{rho,c} (sum_b |A_rho[b,c]|)^2 where
A_rho = [W_rho ; H2_I_rho]  (shape (R, B+1, C)).  Right-multiplying every
slice by a shared orthogonal matrix Q -- A' = A Q -- leaves A' A'^T, hence the
Hamiltonian, error_2 and the angle words, untouched; only the column L1 sums
move.

alexchilton's public note reports -0.2..-0.5% from an LLM-iterated optimiser
(Riemannian conjugate gradient, cached losses, periodic re-orthogonalisation,
multiscale Givens sweeps).  Ours is a re-implementation in our own tree; the
official verifier remains the judge.

Model: f(Q) = 0.25 sum_{rho,c} s[rho,c]^2,  s[rho,c] = sum_b |(A Q)[rho,b,c]|.
Euclidean gradient wrt Q:  E = A^T (0.5 s sign(A Q))  (sum over rho, b).
Riemannian gradient at Q:  P = E - Q sym(Q^T E).  Retraction: polar factor.

Usage:
  python chem_gauge_riemannian.py --npz IN.npz --out OUT.npz [--iters 300]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def l1_objective(a: np.ndarray) -> float:
    return float(0.25 * np.square(np.abs(a).sum(axis=-2)).sum())


def riemannian_cg(a0: np.ndarray, iters: int, tol: float = 1e-13):
    r, _, c = a0.shape
    rng = np.random.default_rng(20261105)
    Q = np.eye(c) + 1e-6 * rng.standard_normal((c, c))
    Q, _ = np.linalg.qr(Q)

    def eg(Qv):
        ap = a0 @ Qv
        s = np.abs(ap).sum(axis=-2)                       # (R, C)
        m = 0.5 * s[..., None, :] * np.sign(ap)           # (R, B+1, C) dF/dAp
        return np.einsum('rbc,rbx->cx', m, a0), ap        # (C, C)

    E, ap = eg(Q)
    P = E - Q @ (Q.T @ E + E.T @ Q) / 2
    D = -P
    f = l1_objective(ap)
    f_prev = f
    for it in range(iters):
        step = 1e-2
        Rn = None
        for _ in range(30):
            U, _, Vt = np.linalg.svd(Q - step * P)
            cand_q = U @ Vt
            fc = l1_objective(a0 @ cand_q)
            if fc < f:
                Rn = cand_q
                break
            step *= 0.5
        if Rn is None:
            break
        Q = Rn
        E, ap = eg(Q)
        P_new = E - Q @ (Q.T @ E + E.T @ Q) / 2
        denom = float(np.vdot(P, P))
        beta = min(1.0, max(0.0, float(np.vdot(P_new, P_new - P)) / denom)) if denom > 0 else 0.0
        D = -P_new + beta * D
        if it % 20 == 0:
            print(f'  it {it:4d} f {f:.9f} step {step:.1e} beta {beta:.3f}', flush=True)
        if f - fc < tol:
            f = fc
            P = P_new
            break
        f, P = fc, P_new
    return Q, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--iters', type=int, default=300)
    args = ap.parse_args()

    with np.load(args.npz, allow_pickle=False) as z:
        data = {k: np.array(z[k]) for k in z.files}

    a0 = np.concatenate([data['W'], data['H2_I'][:, None, :]], axis=1)  # (R, B+1, C)
    f0 = l1_objective(a0)
    t0 = time.time()
    Q, f1 = riemannian_cg(a0, args.iters)
    a1 = a0 @ Q
    defect = float(np.max(np.abs(a1 @ a1.transpose(0, 2, 1) - a0 @ a0.transpose(0, 2, 1))))
    print(f'f {f0:.9f} -> {f1:.9f}  ({100*(f1-f0)/f0:+.5f}%)  gram defect {defect:.2e}  '
          f'{time.time()-t0:.1f}s', flush=True)
    if defect > 1e-9:
        raise ArithmeticError('gauge was not orthogonal')
    if f1 >= f0:
        print('no improvement; not written')
        return
    out = dict(data)
    out['W'], out['H2_I'] = a1[:, :-1, :], a1[:, -1, :]
    if 'offset_W' in out:
        out['offset_W'] = out['H2_I'] - out['W'].sum(axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print('saved', args.out)


if __name__ == '__main__':
    main()

"""Build a (rank, basis) seed for a small-B training run from a larger factorization.

The official train_df_thc_qaggle.load_seed_params takes the FIRST bases of each row when
the target basis count is smaller, which throws away whichever atoms carry the most
weight. Instead, pick the atoms by least-squares value: choose the subset whose rank-1
matrices best reproduce the row's X^(rho c) columns, and refit w by exact least squares.

This does not make B=30 feasible on its own -- a sum of B rank-1 atoms has rank <= B, and
the source X_rc are rank 57-60 -- but it starts the optimizer from the best point that B
atoms can represent, instead of an arbitrary truncation.

Usage: python build_seed.py SEED_NPZ OUT_NPZ --bases 30 --copies 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

N = 76
IU = np.triu_indices(N)
TRI = IU[0] * N + IU[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('seed', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--bases', type=int, default=30)
    ap.add_argument('--copies', type=int, default=30)
    args = ap.parse_args()

    with np.load(args.seed, allow_pickle=False) as a:
        U, W, H2, H1, Hsym = (a['U'], a['W'], a['H2_I'], a['H1_I'], a['Hsym_lower'])
    R, B0, _ = U.shape
    _, _, C0 = W.shape
    B, C = min(args.bases, B0), min(args.copies, C0)
    print(f'seed {R}x{B0}x{C0} -> {R}x{B}x{C}')

    Uo = np.empty((R, B, N), dtype=np.float32)
    Wo = np.empty((R, B, C), dtype=np.float32)
    H2o = np.empty((R, C), dtype=np.float32)
    worst = 0.0
    for rho in range(R):
        u = U[rho].astype(np.float64)
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-30)
        w = W[rho].astype(np.float64)[:, :C]
        X = np.einsum('bp,bq,bc->cpq', u, u, w)          # (C, N, N)
        Xv = X[:, IU[0], IU[1]]                          # (C, 2926)
        G = (u[:, None, :] * u[:, :, None]).reshape(B0, N * N)[:, TRI]
        # Greedy subset selection on the residual, then an exact refit.
        sel: list[int] = []
        resid = Xv
        for _ in range(B):
            coef = G @ resid.T                           # (B0, C) correlation
            score = (coef ** 2).sum(axis=1)
            score[sel] = -1.0
            sel.append(int(np.argmax(score)))
            Z = G[sel]
            wsel, *_ = np.linalg.lstsq(Z.T, Xv.T, rcond=None)
            resid = (Xv.T - Z.T @ wsel).T
        Z = G[sel]
        wsel, *_ = np.linalg.lstsq(Z.T, Xv.T, rcond=None)  # (B, C)
        worst = max(worst, np.linalg.norm((Z.T @ wsel).T - Xv) / np.linalg.norm(Xv))
        Uo[rho] = u[sel].astype(np.float32)
        Wo[rho] = wsel.astype(np.float32)
        H2o[rho] = (H2[rho].astype(np.float64)[:C]
                    + w[:, :C].sum(axis=0) - wsel.sum(axis=0)).astype(np.float32)
    print(f'worst X residual at B={B}: {worst:.3e}')
    np.savez(args.out, U=Uo, W=Wo, H2_I=H2o, H1_I=H1, Hsym_lower=Hsym)
    print('wrote', args.out)


if __name__ == '__main__':
    main()

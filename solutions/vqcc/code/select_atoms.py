"""Drop B from 60 to 30 by selecting the atoms that span each row's own X space.

The lever. T_angles = 2*K*(N + R*B) and chemistry is the only axis that can still move the
composite score. B >= 30 is a hard lower bound (each row's C=30 columns of X are linearly
independent, rank(Xv[rho]) = 30), so B=30, num_vectors = 2,026 is the floor and projects to
a 6.6 composite -- above the ~6.25 top-10% line.

Why it can be exact. For a fixed atom set U, X^(rho c) = U diag(w_c) U^T is LINEAR in w, so
the best weights are a least-squares solve. Each row's 30 X-columns span a 30-dimensional
subspace, and that subspace is carried by some 30 of the row's 60 existing atoms. Selecting
those 30 and refitting w reproduces every X^(rho c) to machine precision, hence error_2
exactly. error_1 is preserved because sum_rc W X is unchanged when we keep w_B fixed.

What can still move is Lambda's weight term, which is whatever the refit gives. The
official evaluator decides. If it holds the gates, this is a real, submittable circuit.

Usage: python select_atoms.py [--bases 30 32 34] [--sub DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401

sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos.verify import load_inputs  # noqa: E402
from dfthcsos.scoreinfra.dfthc_math import (  # noqa: E402
    evaluate_dfthc_vectors,
    reconstruct_u_vectors,
    reconstruct_x,
    split_u_vectors,
)

N = 76
IU = np.triu_indices(N)
TRI = IU[0] * N + IU[1]


def atom_rank1(u_row):
    """The B rank-1 matrices vec(u_b (x) u_b), shape (B, N*N) restricted to the upper triangle."""
    return (u_row[:, None, :] * u_row[:, :, None]).reshape(len(u_row), N * N)[:, TRI]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bases', type=int, nargs='+', default=[30, 32, 34, 40])
    ap.add_argument('--sub', type=Path, default=Path('_scratch/submission/dfthcsos'))
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    inp = load_inputs(args.sub / 'submission.npz')
    P = dict(np.load(args.sub / 'submission.npz', allow_pickle=False))
    brot = int(P['brot']) if 'brot' in P else 10
    v = reconstruct_u_vectors(P['angle_words'] / (1 << (brot - 1)), P['tree_edges'])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    one_body, u = split_u_vectors(v, inp.w)
    R, B0, C = inp.w.shape

    X = reconstruct_x(u, inp.w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    Xv = X[:, :, IU[0], IU[1]]                       # (R, C, 2926)

    def score(uu, ww, WW):
        u_all = np.concatenate([one_body, uu.reshape(uu.shape[0] * uu.shape[1], N)], axis=0)
        return evaluate_dfthc_vectors(u_all, ww, WW, inp.S, inp.beta_1, inp.ref,
                                      energy_upper=inp.energy_upper)

    m0 = score(u, inp.w, inp.W)
    print(f'source B={B0} nv={N + R * B0}: e1 {1e3*m0["error_1"]:.3f} '
          f'e2 {1e3*m0["error_2"]:.3f} mHa Lambda {m0["Lambda"]:.2f} '
          f'leff {m0["lambda_eff"]:.3f}', flush=True)

    best = None
    for basis in args.bases:
        if basis > B0:
            continue
        u_new = np.empty((R, basis, N))
        w_new = np.empty((R, basis, C))
        W_new = np.empty((R, C))
        resid = 0.0
        for rho in range(R):
            Q, _ = np.linalg.qr(Xv[rho].T)          # (2926, 30) basis of the row's X span
            k = Q.shape[1]
            G = atom_rank1(u[rho])                  # (B, 2926)
            coords = Q.T @ G.T                      # (k, B)
            sel = np.argsort(-np.linalg.norm(coords, axis=0))[:basis]
            Z = G[sel]                              # (basis, 2926)
            w, *_ = np.linalg.lstsq(Z.T, Xv[rho].T, rcond=None)   # (basis, C)
            resid = max(resid, np.linalg.norm((Z.T @ w).T - Xv[rho]) / np.linalg.norm(Xv[rho]))
            u_new[rho] = u[rho][sel]
            w_new[rho] = w
            W_new[rho] = inp.W[rho] + inp.w[rho].sum(axis=0) - w.sum(axis=0)

        m = score(u_new, w_new, W_new)
        nv = N + R * basis
        ok = (m['error_1'] <= 0.020 and m['error_2'] <= 0.020 and m['gap_in_domain']
              and m['Lambda'] > 49.0)
        print(f'B={basis:2d} nv={nv:5d}: e1 {1e3*m["error_1"]:9.3f} '
              f'e2 {1e3*m["error_2"]:9.3f} mHa Lambda {m["Lambda"]:8.2f} '
              f'leff {m["lambda_eff"]:8.3f} gate {m["gap_in_domain"]} '
              f'{"FEASIBLE" if ok else "reject"}  x-resid {resid:.2e}', flush=True)
        if ok and (best is None or m['lambda_eff'] < best[0]):
            best = (m['lambda_eff'], basis, u_new.copy(), w_new.copy(), W_new.copy())

    if best is not None and args.out is not None:
        lam, basis, uu, ww, WW = best
        out = dict(P)
        out['U'] = np.concatenate([one_body, uu.reshape(R * basis, N)], axis=0)
        out['W'] = ww
        out['H2_I'] = WW
        args.out.mkdir(parents=True, exist_ok=True)
        np.savez(args.out / 'submission.npz', **out)
        print(f'wrote B={basis} (leff {lam:.4f}) to {args.out}', flush=True)


if __name__ == '__main__':
    main()

"""Compress the DF-THC ensemble index C and lower Lambda's weight term.

The encoded vectors define projectors P_b = u_b u_b^T with Gram matrix G_bd = (u_b . u_d)^2.
Diagonalizing W^T G W exposes the physical ensemble directions; the weakest can be dropped and
the rest realigned to the original sparse columns by orthogonal Procrustes. For each retained
direction, stack A = [w ; w_B] so that Lambda's weight term is 1/4 sum_c ||A_c||_1^2; orthogonal
rotations of A's columns preserve A A^T (hence the two-body Hamiltonian, the absorbed one-body
Hamiltonian and the SOS gap) while changing that one-norm.

Every candidate is scored by the official numpy path before it is accepted.

Usage: python chem_compress.py SUBMISSION_DIR OUT_DIR [--copies C ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401

sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos.verify import (  # noqa: E402
    evaluate_words,
    load_circuits,
    load_inputs,
    rejection_reason,
    resource_costs,
    score_from_metrics,
)
from dfthcsos.scoreinfra.dfthc_math import reconstruct_u_vectors  # noqa: E402


def sparse_rotate(A, *, rounds=3, pairs_per_round=None, seed=619):
    """Orthogonal column Givens rotations that minimize sum_c ||A_c||_1^2.

    Candidate angles include every coordinate-zero event plus a small uniform grid. Each
    accepted rotation preserves A A^T exactly (up to floating arithmetic).
    """
    A = np.asarray(A, float).copy()
    R, D, C = A.shape
    rng = np.random.default_rng(seed)
    if pairs_per_round is None:
        pairs_per_round = 4 * C
    before = float(np.sum(np.abs(A).sum(axis=1) ** 2))
    rr = np.arange(R)
    for _ in range(rounds):
        for _k in range(pairs_per_round):
            a, b = rng.choice(C, 2, replace=False)
            x, y = A[:, :, a].copy(), A[:, :, b].copy()
            events = np.mod(np.arctan2(y, x), np.pi / 2)
            grid = np.broadcast_to(np.linspace(0, np.pi / 2, 9)[None, :], (R, 9))
            angles = np.concatenate([np.zeros((R, 1)), events, grid], axis=1)
            co = np.cos(angles)[:, :, None]
            si = np.sin(angles)[:, :, None]
            values = (np.abs(co * x[:, None, :] + si * y[:, None, :]).sum(axis=2) ** 2
                      + np.abs(co * y[:, None, :] - si * x[:, None, :]).sum(axis=2) ** 2)
            best = np.argmin(values, axis=1)
            old = np.abs(x).sum(1) ** 2 + np.abs(y).sum(1) ** 2
            theta = np.where(values[rr, best] < old - 1e-12, angles[rr, best], 0.0)
            c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
            A[:, :, a] = c * x + s * y
            A[:, :, b] = c * y - s * x
    after = float(np.sum(np.abs(A).sum(axis=1) ** 2))
    return A, {'l1_square_before': before, 'l1_square_after': after}


def compress(params, u, copies, *, rounds=3):
    """Drop to ``copies`` ensemble directions and rotate; returns new params and a log."""
    w = np.asarray(params['W'], float)
    h = np.asarray(params['H2_I'], float)
    gram = (u @ u.transpose(0, 2, 1)) ** 2
    _, q = np.linalg.eigh(w.transpose(0, 2, 1) @ gram @ w)
    q = q[:, :, ::-1][:, :, :copies]
    old = np.concatenate([w, h[:, None, :]], axis=1)          # (R, B+1, C)
    compressed = old @ q
    # Align the compressed factors with the most important original sparse columns.
    significance = np.abs(old).sum(axis=1) ** 2               # (R, C)
    ix = np.argsort(-significance, axis=1)[:, :copies]
    target = np.take_along_axis(old, ix[:, None, :], axis=2)
    U, _, VT = np.linalg.svd(compressed.transpose(0, 2, 1) @ target, full_matrices=False)
    compressed = compressed @ (U @ VT)
    rotated, log = sparse_rotate(compressed, rounds=rounds)
    out = {k: np.array(v, copy=True) for k, v in params.items()}
    out['W'] = rotated[:, :-1, :]
    out['H2_I'] = rotated[:, -1, :]
    return out, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('submission', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--copies', type=int, nargs='+', default=[40, 36, 32, 30, 28])
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--angles-toffoli', type=float, default=None)
    args = ap.parse_args()

    npz = args.submission / 'submission.npz'
    P = dict(np.load(npz, allow_pickle=False))
    inp = load_inputs(npz)
    family = load_circuits(args.submission)
    brot = int(P['brot']) if 'brot' in P else 10
    words, tree = P['angle_words'], P['tree_edges']
    angles_toffoli = args.angles_toffoli
    if angles_toffoli is None:
        angles_toffoli = json.loads((args.submission / 'verify.json').read_text()
                                    )['resources']['angles_toffoli']

    m0 = evaluate_words(inp, words, tree, brot)
    r0 = resource_costs(inp, family, brot, angles_toffoli)
    s0 = score_from_metrics(m0, r0)
    print(f'source: score {s0:.6g} leff {m0["lambda_eff"]:.4f} Q {r0["qubits"]} '
          f'T {r0["toffoli"]} (e1 {1e3*m0["error_1"]:.3f} e2 {1e3*m0["error_2"]:.3f} mHa)',
          flush=True)

    u = reconstruct_u_vectors(words[76:] / (1 << (brot - 1)), tree)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    R, B, C0 = inp.w.shape
    u = u.reshape(R, B, 76)

    best = None
    for copies in args.copies:
        if copies >= C0:
            continue
        params, log = compress(P, u, copies, rounds=args.rounds)
        inp2 = load_inputs(params)
        m = evaluate_words(inp2, words, tree, brot)
        r = resource_costs(inp2, family, brot, angles_toffoli)
        score = score_from_metrics(m, r)
        why = rejection_reason(m, r)
        print(f'C={copies:3d}: score {score:.6g} leff {m["lambda_eff"]:.4f} Q {r["qubits"]} '
              f'T {r["toffoli"]} e1 {1e3*m["error_1"]:.3f} e2 {1e3*m["error_2"]:.3f} mHa '
              f'({why or "accepted"})', flush=True)
        if np.isfinite(score) and (best is None or score < best[0]):
            best = (score, copies, params, m, r)
        del inp2

    if best is None:
        print('no feasible candidate')
        return
    score, copies, params, m, r = best
    print(f'best: C={copies} score {score:.6g} ({s0/score:.4f}x on chemistry)')
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.submission.iterdir():
        if f.name.startswith('angle_k_'):
            (args.out / f.name).write_bytes(f.read_bytes())
    np.savez(args.out / 'submission.npz', **params)
    print('wrote', args.out)


if __name__ == '__main__':
    main()

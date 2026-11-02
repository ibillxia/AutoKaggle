"""L-BFGS-B on the exact lambda_eff, with the analytic gradient through eigh.

Adapted from the public 6.68686 technique (kozykappa/quantum-ai-lb-score-6-68108):
a previous refit here used an Adam penalty method and concluded lambda_eff was pinned.
That was the wrong optimizer. The control variable is the absorbed identity offset
``H2_I`` (equivalently ``offset = H2_I - sum_b w``), and both terms of

    lambda_eff = sqrt(E_gap * (2 Lambda - E_gap))

respond to it: E_gap moves through the eigenvalues of h1_prime, Lambda through both
``sum|eigenvalues|`` and the LCU one-norm ``1/4 sum (|w|_1 + |w_B|)^2``. The exact gradient
of both is available via the derivative of ``eigh``.

The two-body X matrices do NOT change (only the offsets move), so error_2 and the angle
words for the two-body rows are untouched. The one-body rows are re-encoded afterwards.

Usage: python offset_opt.py SUBMISSION_DIR [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401

sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos.scoreinfra import reference as refmod  # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as maths  # noqa: E402
from dfthcsos.verify import load_inputs, evaluate_words  # noqa: E402

GAP_FLOOR = 3.914898
LAMBDA_FLOOR = 49.0


def encode_vectors(vectors, tree_edges, brot):
    work = np.asarray(vectors, float).copy()
    work /= np.linalg.norm(work, axis=1, keepdims=True)
    angles = np.empty((len(work), len(tree_edges)))
    for j in reversed(range(len(tree_edges))):
        p, c = tree_edges[j]
        angles[:, j] = np.arctan2(work[:, c], work[:, p])
        work[:, p] = np.hypot(work[:, p], work[:, c])
        work[:, c] = 0.0
    return np.rint(angles * (2 ** (brot - 1) / np.pi)).astype(np.int64) % (1 << brot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('sub', type=Path)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--maxiter', type=int, default=600)
    ap.add_argument('--gap-margin', type=float, default=1e-3)
    ap.add_argument('--brot', type=int, default=10)
    args = ap.parse_args()

    npz = args.sub / 'submission.npz'
    data = {k: np.array(v, copy=True) for k, v in
            np.load(npz, allow_pickle=False).items()}
    ref = refmod.load_reference(refmod.check_fcidump())
    fixed = refmod.load_fixed_shifts()
    brot = args.brot

    w = np.asarray(data['W'], float)
    n = data['U'].shape[-1]
    shift = np.asarray(maths.symmetric_from_lower(fixed['Hsym_lower'], n), float)
    beta = float(fixed['H1_I']) - 0.5 * float(np.trace(shift))
    sum_w = w.sum(axis=1)
    offset0 = np.asarray(data['H2_I'], float) - sum_w
    r, _, c = w.shape

    words = data['angle_words']
    tree = data['tree_edges']
    vectors = maths.reconstruct_u_vectors(words / (1 << (brot - 1)), tree)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    _, u = maths.split_u_vectors(vectors, w)

    h1_thc = ref['oei'] - 0.5 * np.einsum('prrq->pq', ref['tei'])
    constant = (h1_thc - 0.5 * ref['electrons'] * shift + beta * np.eye(n))
    energy = refmod.UPPER_ENERGY - ref['ecore'] + ref['electrons'] * beta
    l1_w = np.abs(w).sum(axis=1)
    floor = GAP_FLOOR + args.gap_margin

    def quantities(flat):
        off = flat.reshape(r, c)
        comb = np.einsum('rc,rbc->rb', off, w)
        h1 = constant - np.einsum('rb,rbp,rbq->pq', comb, u, u, optimize=True)
        vals, vecs = np.linalg.eigh(h1)
        sign = (vecs * np.sign(vals)) @ vecs.T
        contr = np.einsum('rbp,pq,rbq->rb', u, sign, u, optimize=True)
        gap = float(energy + np.abs(vals).sum() - vals.sum()
                    + 0.5 * np.sum(off ** 2))
        w_b = off + sum_w
        lam = float(np.abs(vals).sum() + 0.25 * np.sum((l1_w + np.abs(w_b)) ** 2))
        g_gap = off + np.einsum('rb,rbc->rc', 1.0 - contr, w)
        g_lam = (-np.einsum('rb,rbc->rc', contr, w)
                 + 0.5 * (l1_w + np.abs(w_b)) * np.sign(w_b))
        return gap, lam, g_gap, g_lam

    def objective(flat):
        gap, lam, gg, gl = quantities(flat)
        rad = gap * (2.0 * lam - gap)
        if rad <= 0:
            return 1e12 + abs(rad), np.zeros_like(flat)
        loss = 0.5 * math.log(rad)
        grad = ((lam - gap) * gg + gap * gl) / rad
        if gap < floor:
            loss += 1e5 * (floor - gap) ** 2
            grad = grad - 2e5 * (floor - gap) * gg
        if lam < LAMBDA_FLOOR + 1e-3:
            loss += 1e5 * (LAMBDA_FLOOR + 1e-3 - lam) ** 2
            grad = grad - 2e5 * (LAMBDA_FLOOR + 1e-3 - lam) * gl
        return loss, grad.ravel()

    gap0, lam0, _, _ = quantities(offset0.ravel())
    before = math.sqrt(gap0 * (2 * lam0 - gap0))
    print(f'before: gap {gap0:.6f}  Lambda {lam0:.4f}  lambda_eff {before:.6f}', flush=True)

    res = minimize(objective, offset0.ravel(), jac=True, method='L-BFGS-B',
                   options={'maxiter': args.maxiter, 'maxcor': 20,
                            'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 60})
    off = res.x.reshape(r, c)
    gap1, lam1, _, _ = quantities(off.ravel())
    after = math.sqrt(gap1 * (2 * lam1 - gap1))
    print(f'after : gap {gap1:.6f}  Lambda {lam1:.4f}  lambda_eff {after:.6f}  '
          f'({before / after:.4f}x)  nit={res.nit} success={res.success}', flush=True)

    if not (gap1 > GAP_FLOOR and lam1 > LAMBDA_FLOOR):
        print('REJECTED: floors not satisfied', flush=True)
        return

    out = {k: np.array(v, copy=True) for k, v in data.items()}
    out['H2_I'] = (off + sum_w).astype(np.float32) if data['H2_I'].dtype == np.float32 \
        else off + sum_w
    # Re-encode the one-body rows from the new h1_prime eigenvectors.
    comb = np.einsum('rc,rbc->rb', off, w)
    h1 = constant - np.einsum('rb,rbp,rbq->pq', comb, u, u, optimize=True)
    _, vecs = np.linalg.eigh(h1)
    out['angle_words'][:n] = encode_vectors(vecs.T, tree, brot)

    m = evaluate_words(load_inputs(out), out['angle_words'], tree, brot)
    print(f'official words: e1 {1e3*m["error_1"]:.3f}  e2 {1e3*m["error_2"]:.3f} mHa  '
          f'Lambda {m["Lambda"]:.3f}  E_gap {m["E_gap"]:.4f}  '
          f'lambda_eff {m["lambda_eff"]:.4f}', flush=True)
    ok = (m['error_1'] <= 0.020 and m['error_2'] <= 0.020 and m['Lambda'] > 49.0
          and m['E_gap'] > GAP_FLOOR and m['gap_in_domain'])
    print('PASS' if ok else 'FAIL', flush=True)
    if ok and args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        np.savez(args.out / 'submission.npz', **out)
        print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()

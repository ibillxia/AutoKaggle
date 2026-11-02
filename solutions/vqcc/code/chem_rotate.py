"""Lower Lambda by rotating the ensemble index c, which leaves the accuracy metrics invariant.

With w (R,B,C) and w_B (R,C), the two-body sum is X^(rc) = sum_b w_b u_b u_b^T. Rotating the C
ensemble columns of one rho by an orthogonal Q gives w' = w Q and w_B' = w_B Q, and since the
c index never couples different rho:

  sum_c X^(rc) (x) X^(rc) = sum_b (w w^T)_bb' u_b u_b'^T   is unchanged   -> error_2 fixed
  W'^(rc) X'^(rc) = W^(rc) X^(rc)                          is unchanged   -> error_1 fixed
  sum_c W'^2 = W W^T                                       is unchanged   -> E_SOS fixed

Only Lambda's weight term  0.25 sum_rc (sum_b |w'| + |w_B'|)^2  moves, so lambda_eff moves with
it. This is a free reparametrisation: the circuits and the loaded vectors are untouched.

Each rho is independent: minimize  f_c(w') = (sum_b |w'_bc| + |w'_Bc|)^2  for c = 1..C, where
w'_bc = sum_d Q_cd w_bd is the rotated column. Rotations are parametrized in the Cayley form
Q = (I - A)(I + A)^-1 with A skew-symmetric, which keeps every step orthogonal by construction.

Usage: python chem_rotate.py SUBMISSION_DIR OUT_DIR [--iters N] [--time SEC]
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


def weight_term(w, wB):
    """Lambda's second term: 0.25 sum_rc (sum_b |w_b| + |w_B|)^2."""
    return 0.25 * ((np.abs(w).sum(axis=1) + np.abs(wB)) ** 2).sum()


def cayley(A):
    """Orthogonal Q = (I - A)(I + A)^-1 for skew-symmetric A."""
    n = A.shape[-1]
    eye = np.eye(n)
    return (eye - A) @ np.linalg.inv(eye + A)


def per_rho_objective(w, wB):
    """(R, C) array of the per-rho, per-column terms of the weight term."""
    return 0.25 * ((np.abs(w).sum(axis=1) + np.abs(wB)) ** 2)


def optimize(w, wB, *, iters=4000, lr=0.02, seed=0, log_every=500):
    """Gradient descent on the Cayley parameters, one independent A per rho.

    d/dA of the objective is taken numerically through a finite-difference-free path: the
    columns of Q are the only things that enter, so we use torch autograd when available and
    otherwise a simple coordinate search over Givens angles.
    """
    import torch

    torch.set_default_dtype(torch.float64)
    R, B, C = w.shape
    wt = torch.as_tensor(w)
    wBt = torch.as_tensor(wB)
    A = torch.zeros((R, C, C), dtype=torch.float64, requires_grad=True)
    eye = torch.eye(C, dtype=torch.float64)
    opt = torch.optim.Adam([A], lr=lr)
    best, best_Q = float('inf'), None

    def objective(Ap):
        As = Ap - Ap.transpose(1, 2)                       # skew-symmetric
        Q = (eye - As) @ torch.linalg.inv(eye + As)        # (R, C, C)
        wr = torch.einsum('rbc,rcd->rbd', wt, Q)           # (R, B, C)
        wBr = torch.einsum('rc,rcd->rd', wBt, Q)           # (R, C)
        return (0.25 * ((wr.abs().sum(dim=1) + wBr.abs()) ** 2)).sum()

    for it in range(iters + 1):
        opt.zero_grad()
        loss = objective(A)
        v = float(loss.detach())
        if v < best:
            best = v
            As = (A.detach() - A.detach().transpose(1, 2))
            best_Q = ((eye - As) @ torch.linalg.inv(eye + As)).numpy()
        if it % log_every == 0:
            print(f'  iter {it:5d}: weight term {v:.6f}  (best {best:.6f})', flush=True)
        if it == iters:
            break
        loss.backward()
        opt.step()
    return best_Q, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('submission', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=0.02)
    args = ap.parse_args()

    npz = args.submission / 'submission.npz'
    P = dict(np.load(npz, allow_pickle=False))
    inp = load_inputs(npz)
    family = load_circuits(args.submission)
    brot = int(P['brot']) if 'brot' in P else 10
    words, tree = P['angle_words'], P['tree_edges']
    rep = json.loads((args.submission / 'verify.json').read_text())
    angles_toffoli = rep['resources']['angles_toffoli']

    m0 = evaluate_words(inp, words, tree, brot)
    s0 = score_from_metrics(m0, resource_costs(inp, family, brot, angles_toffoli))
    print(f'source: score {s0:.6g} leff {m0["lambda_eff"]:.4f} Lambda {m0["Lambda"]:.4f}', flush=True)

    wB = inp.W + inp.w.sum(axis=1)
    print(f'start weight term {weight_term(inp.w, wB):.6f}', flush=True)
    Q, best = optimize(inp.w, wB, iters=args.iters, lr=args.lr)

    w2 = np.einsum('rbc,rcd->rbd', inp.w, Q)
    wB2 = np.einsum('rc,rcd->rd', wB, Q)
    params = {**P, 'W': w2, 'H2_I': wB2}
    inp2 = load_inputs(params)
    m = evaluate_words(inp2, words, tree, brot)
    r = resource_costs(inp2, family, brot, angles_toffoli)
    score = score_from_metrics(m, r)
    why = rejection_reason(m, r)
    print(f'rotated: score {score:.6g} leff {m["lambda_eff"]:.4f} Lambda {m["Lambda"]:.4f} '
          f'e1 {1e3*m["error_1"]:.6f} e2 {1e3*m["error_2"]:.6f} mHa ({why or "accepted"})')
    if not np.isfinite(score) or score >= s0:
        print('no gain; not written')
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.submission.iterdir():
        if f.name.startswith('angle_k_'):
            (args.out / f.name).write_bytes(f.read_bytes())
    np.savez(args.out / 'submission.npz', **params)
    print(f'wrote {args.out}  ({s0/score:.4f}x on the chemistry cost)')


if __name__ == '__main__':
    main()

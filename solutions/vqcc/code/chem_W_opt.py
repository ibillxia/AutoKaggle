"""Optimise W (the one-body absorption matrix) to minimise lambda_eff.

The verifier treats W_rc as a free (R, C) matrix: it enters h1_prime
(-sum_rc W X), E_SOS (-0.5||W||^2) and lambda_h2 (through H2_I = W + sum_b w_b).
So W is not "determined" by the factorization -- it is a free parameter of size
R*C = 1950, and the submitted values are only one choice.

Sanity-checked against the official evaluator before optimising. The reported
metrics here must reproduce evaluate_words() exactly or the run aborts.

Usage: python chem_W_opt.py [--iters N] [--lr X]
"""
from __future__ import annotations

import argparse
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

import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

E_UPPER = -22140.27335
LAM_FLOOR, GAP_FLOOR = 49.0, 3.914898
ERR1_LIMIT = 20e-3  # the verifier gate is 20 mHa; do not tighten it


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--npz', type=Path,
                    default=env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz')
    a = ap.parse_args()

    z = np.load(a.npz)
    inp = V.load_inputs(a.npz)
    X = M.reconstruct_x(z['U'], inp.w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)

    num_orbitals = 76
    eta = inp.ref['electrons']
    # h1_prime = h1_thc - (eta/2) S + beta_1 I - sum_rc W_rc X_rc   (Eq. 38)
    H0 = (inp.ref['h1_thc'] - 0.5 * eta * inp.S
          + inp.beta_1 * np.eye(num_orbitals))
    A = inp.ref['ecore'] - eta * inp.beta_1
    s_np = inp.w.sum(axis=1)
    l1_np = np.abs(inp.w).sum(axis=1)

    H0t = torch.tensor(H0)
    Xt = torch.tensor(X)
    sT = torch.tensor(s_np)
    l1T = torch.tensor(l1_np)
    rank_r, copies_c = s_np.shape
    # The N one-body rows the circuits load, rebuilt from the untouched word
    # table. These are constants of the W problem.
    _u_all = M.reconstruct_u_vectors(z['angle_words'] / (1 << 9), z['tree_edges'])
    _one_body, _ = M.split_u_vectors(_u_all, inp.w)
    VT = torch.tensor(_one_body / np.linalg.norm(_one_body, axis=1, keepdims=True))

    def evaluate(Wv: torch.Tensor):
        W = Wv.reshape(rank_r, copies_c)
        h1 = H0t - torch.einsum('rc,rcpq->pq', W, Xt)
        h1 = 0.5 * (h1 + h1.T)
        ev = torch.linalg.eigvalsh(h1)
        lam_h1 = ev.abs().sum()
        h2_i = W + sT
        lam_h2 = 0.25 * ((h2_i.abs() + l1T) ** 2).sum()
        lam = lam_h1 + lam_h2
        e_sos = A - (lam_h1 - ev.sum()) - 0.5 * (W ** 2).sum()
        e_gap = E_UPPER - e_sos
        l_eff = torch.sqrt(torch.clamp(e_gap * (2.0 * lam - e_gap), min=0.0))
        # error_1 = ||sum_j lambda_j v_j v_j^T - h1_prime||_F, where v_j are the
        # one-body rows the (untouched) circuits load. Changing W moves
        # h1_prime AND its eigenvalues away from the operator those fixed rows
        # encode. This gate is what actually binds W: without it the optimiser
        # buys Lambda with error_1.
        recon = torch.einsum('j,jp,jq->pq', ev, VT, VT)
        err1 = torch.linalg.norm(recon - h1)
        return l_eff, lam, e_gap, err1

    m = V.evaluate_words(inp, z['angle_words'], z['tree_edges'], 10)
    Wv = torch.tensor(inp.W.ravel().copy(), requires_grad=True)
    le, lam, eg, e1 = evaluate(Wv)
    print(f"sanity: lam_eff {le.item():.9f} vs {m['lambda_eff']:.9f}")
    print(f"        Lambda  {lam.item():.9f} vs {m['Lambda']:.9f}")
    print(f"        E_gap   {eg.item():.9f} vs {m['E_gap']:.9f}")
    print(f"        error_1 {e1.item():.9f} vs {m['error_1']:.9f}")
    for name, got, want in (('lam_eff', le.item(), m['lambda_eff']),
                            ('Lambda', lam.item(), m['Lambda']),
                            ('E_gap', eg.item(), m['E_gap']),
                            ('error_1', e1.item(), m['error_1'])):
        if not math.isclose(got, want, rel_tol=1e-7, abs_tol=1e-12):
            raise SystemExit(f'{name} mismatch: the model does not reproduce the verifier')
    print('  -> model reproduces the official evaluator\n')

    best = (le.item(), Wv.detach().clone())
    opt = torch.optim.Adam([Wv], lr=a.lr)
    for it in range(a.iters + 1):
        opt.zero_grad()
        le, lam, eg, e1 = evaluate(Wv)
        # Hard margin: a candidate is only feasible if E_gap clears the strict
        # floor by a real margin, so the optimum is not left sitting on it.
        pen = (torch.relu(GAP_FLOOR + 5e-3 - eg) + torch.relu(LAM_FLOOR + 1e-2 - lam)) * 1e3
        pen = pen + torch.relu(e1 - 0.9 * ERR1_LIMIT) * 1e3
        (le + pen).backward()
        opt.step()
        if it % 500 == 0:
            print(f'  it {it:4d}  lam_eff {le.item():10.6f}  Lambda {lam.item():9.4f} '
                  f' E_gap {eg.item():8.6f}  err1 {1e3 * e1.item():8.2f} mHa  pen {pen.item():.3f}')
        if (pen.item() == 0.0 and eg.item() > GAP_FLOOR
                and e1.item() <= ERR1_LIMIT and le.item() < best[0]):
            best = (le.item(), Wv.detach().clone())

    le, lam, eg, e1 = evaluate(best[1])
    print()
    print(f'BEST lam_eff {le.item():.6f}  (was {m["lambda_eff"]:.6f})  '
          f'x{m["lambda_eff"] / le.item():.6f} on the chemistry ratio')
    print(f'  Lambda {lam.item():.4f} (was {m["Lambda"]:.4f})')
    print(f'  E_gap  {eg.item():.6f} (was {m["E_gap"]:.6f})')
    W_best = best[1].numpy().reshape(rank_r, copies_c)
    out = env.SCRATCH / 'W_opt_final.npz'
    np.savez(out, W=W_best)
    print(f'  saved -> {out}')
    rep = dict(lambda_eff=le.item(), Lambda=lam.item(), E_gap=eg.item(),
               lambda_eff_before=m['lambda_eff'], Lambda_before=m['Lambda'],
               error_1=float(e1.item()), error_1_before=m['error_1'],
               ratio_gain=m['lambda_eff'] / le.item())
    (env.SCRATCH / 'W_opt_final.json').write_text(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()

"""Optimize the submitted H2_I (equivalently W_rc) to minimize lambda_eff.

The verifier freezes beta_1 and S but reads W_rc = H2_I - sum_b w_b from the
submitted array, and W_rc enters

    h1_prime = C0 - sum_rc W_rc X_rc            (C0 = h1_thc - (eta/2)S + beta_1 I)
    lambda_h1 = sum_j |lambda_j(h1_prime)|
    lambda_h2 = 0.25 sum_rc (|W_rc + s_rc| + L1_rc)^2,   s = sum_b w
    E_SOS     = ecore - eta*beta_1 - (lambda_h1 - sum lambda) - 0.5 ||W||^2
    E_gap     = E_upper - E_SOS     (must be > 3.914898)
    lambda_eff = sqrt(E_gap (2 Lambda - E_gap)),  Lambda = lambda_h1 + lambda_h2

X_rc is fixed by the two-body rows, so this is a smooth-ish optimization in
1950 variables against a 2926-dimensional linear map. The one-body rows can be
re-encoded as eigenvectors of the new h1_prime afterwards, which makes
error_1 = 0 exactly.

Usage: python opt_W.py [--steps N] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as M  # noqa: E402

E_UPPER = -22140.27335
FLOOR_GAP = 3.914898
FLOOR_LAM = 49.0


def load_problem(npz: Path):
    z = np.load(npz)
    inp = V.load_inputs(npz)
    ref, w = inp.ref, inp.w
    N = int(ref['orbitals'])
    vectors = M.reconstruct_u_vectors(
        np.asarray(z['angle_words'], dtype=np.float64) / 2.0 ** 9, z['tree_edges'])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    _one, u_b_rho = M.split_u_vectors(vectors, w)
    X = M.reconstruct_x(u_b_rho, w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    C0 = ref['h1_thc'] - 0.5 * int(ref['electrons']) * inp.S + inp.beta_1 * np.eye(N)
    return dict(z=z, inp=inp, ref=ref, w=w, X=X, C0=C0, N=N,
                ecore=ref['ecore'], eta=int(ref['electrons']),
                beta_1=inp.beta_1, S=inp.S,
                L1=np.linalg.norm(w, ord=1, axis=1), s=w.sum(axis=1))


def evaluate_W(P, W: np.ndarray) -> dict:
    """The verifier's own Lambda / E_gap / lambda_eff for a given W_rc."""
    h1p = P['C0'] - np.einsum('rc,rcpq->pq', W, P['X'])
    ev = np.linalg.eigvalsh(h1p)
    lam_h1 = float(np.abs(ev).sum())
    H2 = W + P['s']
    lam_h2 = 0.25 * float(np.sum((np.abs(H2) + P['L1']) ** 2))
    Lam = lam_h1 + lam_h2
    E_SOS = (P['ecore'] - P['eta'] * P['beta_1']
             - (lam_h1 - float(ev.sum())) - 0.5 * float(np.sum(W ** 2)))
    E_gap = E_UPPER - E_SOS
    ok = 0.0 < E_gap <= 2 * Lam and E_gap > FLOOR_GAP and Lam > FLOOR_LAM
    le = math.sqrt(E_gap * (2 * Lam - E_gap)) if ok else float('nan')
    return dict(lambda_h1=lam_h1, lambda_h2=lam_h2, Lambda=Lam, E_gap=E_gap,
                E_SOS=E_SOS, lambda_eff=le, ok=bool(ok),
                eig_min=float(ev.min()), eig_max=float(ev.max()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--npz', type=Path,
                    default=env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz')
    ap.add_argument('--out', type=Path, default=env.SCRATCH / 'W_opt_best.npz')
    a = ap.parse_args()

    P = load_problem(a.npz)
    cur = evaluate_W(P, P['inp'].W)
    print('start:', json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                for k, v in cur.items()}))
    print(f"lambda_h2 floor at fixed w: 0.25*sum L1^2 = {0.25*float(np.sum(P['L1']**2)):.4f}")

    Xt = torch.tensor(P['X'], dtype=torch.float64)          # (R, C, N, N)
    C0t = torch.tensor(P['C0'], dtype=torch.float64)
    st = torch.tensor(P['s'], dtype=torch.float64)
    L1t = torch.tensor(P['L1'], dtype=torch.float64)

    # Start from the submitted W; the problem is non-convex so the start matters.
    Wv = torch.tensor(P['inp'].W, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([Wv], lr=a.lr)
    best = (float('inf'), None, None)
    for step in range(a.steps + 1):
        opt.zero_grad()
        h1p = C0t - torch.einsum('rc,rcpq->pq', Wv, Xt)
        ev = torch.linalg.eigvalsh(h1p)
        lam_h1 = torch.sqrt(ev ** 2 + 1e-12).sum()
        H2 = Wv + st
        lam_h2 = 0.25 * ((torch.sqrt(H2 ** 2 + 1e-12) + L1t) ** 2).sum()
        Lam = lam_h1 + lam_h2
        E_SOS = (P['ecore'] - P['eta'] * P['beta_1']
                 - (lam_h1 - ev.sum()) - 0.5 * (Wv ** 2).sum())
        E_gap = E_UPPER - E_SOS
        # We want E_gap just above the floor and Lambda as small as possible.
        gap_pen = 1e4 * torch.relu(FLOOR_GAP * 1.0005 - E_gap) ** 2
        lam_pen = 1e4 * torch.relu(FLOOR_LAM * 1.0005 - Lam) ** 2
        loss = Lam + gap_pen + lam_pen
        loss.backward()
        opt.step()
        if step % 200 == 0:
            Wn = Wv.detach().numpy()
            r = evaluate_W(P, Wn)
            le = r['lambda_eff']
            if r['ok'] and le < best[0]:
                best = (le, Wn.copy(), r)
            print(f"step {step:5d} loss {loss.item():12.4f} Lam {float(Lam):9.3f} "
                  f"Eg {float(E_gap):9.4f} lam_eff {le:9.4f}", flush=True)

    if best[1] is None:
        print('no feasible point found')
        return
    le, Wn, r = best
    print()
    print('BEST:', json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                               for k, v in r.items()}))
    print(f"lambda_eff {47.85036566452195:.4f} -> {le:.4f}  = "
          f"{47.85036566452195 / le:.3f}x")
    np.savez(a.out, W=Wn, H2_I=Wn + P['s'], metrics=json.dumps(r))
    print('wrote', a.out)


if __name__ == '__main__':
    main()

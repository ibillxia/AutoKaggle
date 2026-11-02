"""Optimize W_rc for a factorization, minimizing Lambda subject to the E_gap floor.

The verifier's constraints reduce to one inequality. With
A = ecore - eta*beta_1 and E_SOS = A - (lambda_h1 - sum lambda) - 0.5||W||^2,

    E_gap = E_upper - E_SOS > FLOOR_GAP
  <=> (lambda_h1 - sum lambda) + 0.5||W||^2 > NEED,   NEED = A - E_upper + FLOOR_GAP

and the objective is Lambda = lambda_h1 + lambda_h2. The one-body rows can be
re-encoded as eigenvectors of the resulting h1_prime, which makes error_1 = 0
exactly, so this objective is the whole story for a fixed factorization.

Usage: python opt_W2.py CKPT.npz [--steps N] [--lr X]
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


def prepare(ckpt: Path, words=None, edges=None):
    z = np.load(ckpt)
    base = {k: z[k] for k in z.files}
    inp = V.load_inputs(base)
    w, ref = inp.w, inp.ref
    N, eta = int(ref['orbitals']), int(ref['electrons'])
    # Two-body rows: the checkpoint's U if present, else decode the circuits' words.
    if 'U' in z.files:
        U = np.asarray(z['U'], dtype=np.float64)
    else:
        v = M.reconstruct_u_vectors(np.asarray(words, float) / 2.0 ** 9, edges)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        _one, U = M.split_u_vectors(v, w)
    X = M.reconstruct_x(U, w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    C0 = ref['h1_thc'] - 0.5 * eta * inp.S + inp.beta_1 * np.eye(N)
    return dict(w=w, X=X, C0=C0, N=N, eta=eta, ecore=ref['ecore'],
                beta_1=inp.beta_1, s=w.sum(axis=1),
                L1=np.linalg.norm(w, ord=1, axis=1), W0=inp.W,
                A=ref['ecore'] - eta * inp.beta_1)


def report(P, W: np.ndarray) -> dict:
    h1p = P['C0'] - np.einsum('rc,rcpq->pq', W, P['X'])
    ev = np.linalg.eigvalsh(h1p)
    l1 = float(np.abs(ev).sum())
    l2 = 0.25 * float(np.sum((np.abs(W + P['s']) + P['L1']) ** 2))
    Lam = l1 + l2
    E_SOS = P['A'] - l1 + float(ev.sum()) - 0.5 * float(np.sum(W ** 2))
    Eg = E_UPPER - E_SOS
    ok = 0.0 < Eg <= 2 * Lam and Eg > FLOOR_GAP and Lam > FLOOR_LAM
    return dict(lambda_h1=l1, lambda_h2=l2, Lambda=Lam, E_gap=Eg,
                need=l1 - float(ev.sum()) + 0.5 * float(np.sum(W ** 2)),
                lambda_eff=math.sqrt(Eg * (2 * Lam - Eg)) if ok else float('nan'),
                ok=bool(ok))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', type=Path)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--out', type=Path, default=env.SCRATCH / 'W_opt2.npz')
    a = ap.parse_args()

    P = prepare(a.ckpt)
    need = P['A'] - E_UPPER + FLOOR_GAP
    print('lambda_h2 floor 0.25 sum L1^2 =', 0.25 * float(np.sum(P['L1'] ** 2)))
    print('NEED =', need)
    r0 = report(P, P['W0'])
    print('as trained:', json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                     for k, v in r0.items()}))

    Xt = torch.tensor(P['X'], dtype=torch.float64)
    C0t = torch.tensor(P['C0'], dtype=torch.float64)
    st = torch.tensor(P['s'], dtype=torch.float64)
    L1t = torch.tensor(P['L1'], dtype=torch.float64)
    Wv = torch.tensor(P['W0'], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([Wv], lr=a.lr)
    best = (float('inf'), None, None)
    for step in range(a.steps + 1):
        opt.zero_grad()
        h1p = C0t - torch.einsum('rc,rcpq->pq', Wv, Xt)
        ev = torch.linalg.eigvalsh(h1p)
        l1 = torch.sqrt(ev ** 2 + 1e-12).sum()
        l2 = 0.25 * ((torch.sqrt((Wv + st) ** 2 + 1e-12) + L1t) ** 2).sum()
        Lam = l1 + l2
        n = l1 - ev.sum() + 0.5 * (Wv ** 2).sum()
        loss = Lam + 1e3 * torch.relu(need * 1.002 - n) ** 2 + \
            1e3 * torch.relu(FLOOR_LAM * 1.002 - Lam) ** 2
        loss.backward()
        opt.step()
        if step % 250 == 0:
            Wn = Wv.detach().numpy()
            r = report(P, Wn)
            if r['ok'] and r['lambda_eff'] < best[0]:
                best = (r['lambda_eff'], Wn.copy(), r)
            print(f"step {step:5d} loss {float(loss):11.4f} Lam {float(Lam):9.3f} "
                  f"need {float(n):9.3f} lam_eff {r['lambda_eff']:9.4f}", flush=True)
    if best[1] is None:
        print('no feasible point')
        return
    le, Wn, r = best
    print('\nBEST:', json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in r.items()}))
    print(f'47.8504 -> {le:.4f} = {47.85036566452195 / le:.3f}x')
    np.savez(a.out, W=Wn, H2_I=Wn + P['s'], metrics=json.dumps(r))


if __name__ == '__main__':
    main()

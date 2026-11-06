"""Joint refit of the DF-THC coefficient tensor `w` (R,B,C) and the native
weights `Wn` (R,C), with the words and tree fixed.

Why
---
Every earlier refit in this tree moved only `Wn` (`chem_native_w.py`), only the
gauge (`chem_gauge_*.py`), or only inside the active constraint surface
(`chem_constrained_descent.py`).  The coefficient tensor itself was held
fixed.  That leaves one direction unused, and it is the one that trades
*two-body* accuracy for `lambda_eff`:

    X_rc     = sum_b w_rbc u_rb u_rb^T
    a_rb     = sum_c Wn_rc w_rbc
    h1'      = M0 - sum_rb a_rb u_rb u_rb^T
    Lambda   = sum_j |l_j| + 1/4 sum_rc (sum_b |w_rbc| + |Wn_rc + sum_b w_rbc|)^2
    E_gap    = E_upper - (ecore - eta beta - sum(|l_j| - l_j) - 1/2 sum Wn^2)
    lambda_eff = sqrt(E_gap (2 Lambda - E_gap))

`error_2` depends on `w` only, through the per-group Grams `G_rho = w_rho w_rho^T`,
so it can be evaluated exactly from the reduced model

    error_2^2 = || Rm blockdiag(G) Rm^T - Q^T T Q ||^2 + const

with `M = QR`, `M[(pq),(rho,b)] = u_rb[p] u_rb[q]`.  That makes the constraint
cheap enough to carry in an augmented Lagrangian.

Usage:
  python chem_joint_refit.py --npz IN.npz --out OUT.npz [--outer 40] [--inner 400]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as dm  # noqa: E402

ERR1 = V.ERROR_1_THRESHOLD
ERR2 = V.ERROR_2_THRESHOLD
GAPF = V.E_GAP_FLOOR


class Joint:
    def __init__(self, f: dict):
        inp = V.load_inputs(f)
        words = np.asarray(f['angle_words'])
        edges = np.asarray(f['tree_edges'])
        brot = int(f['brot']) if 'brot' in f else 8
        u = dm.reconstruct_u_vectors(words / (1 << (brot - 1)), edges)
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        n = inp.num_orbitals
        self.R, self.B, self.C = inp.w.shape
        self.n = n
        self.ub = u[n:].reshape(self.R, self.B, n)
        m = np.einsum('tbp,tbq->pqtb', self.ub, self.ub).reshape(n * n, self.R * self.B)
        qm, rm = np.linalg.qr(m)
        tei = np.asarray(inp.ref['tei'], float).reshape(n * n, n * n)
        s = np.asarray(inp.S, float)
        eye = np.eye(n).ravel()
        sv = s.ravel()
        t = tei + 0.5 * (np.outer(sv, eye) + np.outer(eye, sv))
        self.Rm, self.Tt = rm, qm.T @ t @ qm
        self.const2 = float(np.sum(t * t) - np.sum(self.Tt * self.Tt))
        ref = inp.ref
        self.eta, self.ecore = ref['electrons'], ref['ecore']
        self.e_upper, self.beta = inp.energy_upper, inp.beta_1
        h1 = ref['oei'] - 0.5 * np.einsum('prrq->pq', ref['tei'])
        self.m0 = h1 - 0.5 * self.eta * inp.S + self.beta * np.eye(h1.shape[0])
        self.one_body = np.asarray(dm.split_u_vectors(u, inp.w)[0], float)
        self.w0 = np.array(inp.w, float)
        self.wn0 = np.array(inp.W, float)
        self.nw = self.R * self.B * self.C

    def split(self, v):
        return (v[:self.nw].reshape(self.R, self.B, self.C),
                v[self.nw:].reshape(self.R, self.C))

    def e2(self, w):
        r, b = self.R, self.B
        g = np.einsum('rbc,rdc->rbd', w, w)
        gb = np.zeros((r * b,) * 2)
        for i in range(r):
            gb[i * b:(i + 1) * b, i * b:(i + 1) * b] = g[i]
        y = self.Rm @ gb @ self.Rm.T - self.Tt
        return float(np.sum(y * y)) + self.const2

    def value_grad(self, v, need_grad=True):
        w, z = self.split(v)
        a = np.einsum('rc,rbc->rb', z, w)
        h1p = self.m0 - np.einsum('rb,rbp,rbq->pq', a, self.ub, self.ub, optimize=True)
        ev, vec = np.linalg.eigh(h1p)
        sg = np.sign(ev)
        lam1 = float(np.abs(ev).sum())
        wl1 = np.abs(w).sum(axis=1)
        wsm = w.sum(axis=1)
        t = wl1 + np.abs(z + wsm)
        lam2 = float(0.25 * np.square(t).sum())
        big = lam1 + lam2
        neg = float((np.abs(ev) - ev).sum())
        e_sos = self.ecore - self.eta * self.beta - neg - 0.5 * float(np.square(z).sum())
        gap = self.e_upper - e_sos
        rad = gap * (2 * big - gap)
        if rad <= 0:
            return float('inf'), None, gap
        val = float(np.sqrt(rad))
        if not need_grad:
            return val, None, gap
        proj = np.square(np.einsum('rbp,pj->rbj', self.ub, vec, optimize=True))
        pb = np.einsum('rbj,j->rb', proj, sg)
        pn = np.einsum('rbj,j->rb', proj, sg - 1.0)
        d_big_w = (-np.einsum('rc,rb->rbc', z, pb)
                   + 0.5 * t[:, None, :] * (np.sign(w) + np.sign(z + wsm)[:, None, :]))
        d_big_z = -np.einsum('rbc,rb->rc', w, pb) + 0.5 * t * np.sign(z + wsm)
        d_neg_w = -np.einsum('rc,rb->rbc', z, pn)
        d_neg_z = -np.einsum('rbc,rb->rc', w, pn)
        d_val_w = (d_neg_w * (big - gap) + gap * d_big_w) / val
        d_val_z = ((d_neg_z + z) * (big - gap) + gap * d_big_z) / val
        return val, np.concatenate([d_val_w.ravel(), d_val_z.ravel()]), gap

    def metrics(self, v):
        w, z = self.split(v)
        lam, _g, gap = self.value_grad(v, need_grad=False)
        a = np.einsum('rc,rbc->rb', z, w)
        h1p = self.m0 - np.einsum('rb,rbp,rbq->pq', a, self.ub, self.ub, optimize=True)
        ev, vec = np.linalg.eigh(h1p)
        rows, cols = linear_sum_assignment(-np.abs(self.one_body @ vec))
        matched = np.empty_like(ev)
        matched[rows] = ev[cols]
        recon = np.einsum('j,jp,jq->pq', matched, self.one_body, self.one_body, optimize=True)
        return dict(lam_eff=lam, err1=float(np.linalg.norm(recon - h1p)),
                    err2=float(np.sqrt(max(self.e2(w), 0.0))), gap=gap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--outer', type=int, default=40)
    ap.add_argument('--inner', type=int, default=400)
    ap.add_argument('--mu0', type=float, default=1e3)
    args = ap.parse_args()

    with np.load(args.npz, allow_pickle=False) as zf:
        f = {k: np.array(zf[k]) for k in zf.files}
    j = Joint(f)
    v0 = np.concatenate([j.w0.ravel(), j.wn0.ravel()])
    m0 = j.metrics(v0)
    print('start', json.dumps(m0), flush=True)
    e2cap = min(ERR2, m0['err2']) - 1e-9
    gapcap = max(GAPF, m0['gap']) + 2e-8
    v = v0.copy()
    mu = args.mu0
    t0 = time.time()
    best = (m0['lam_eff'], v0.copy())

    for outer in range(args.outer):
        def obj(x):
            val, g, gap = j.value_grad(x)
            w, _ = j.split(x)
            e2 = j.e2(w)
            pen = mu * max(0.0, e2 - e2cap) ** 2 + mu * max(0.0, gapcap - gap) ** 2
            return val + pen, g

        res = minimize(obj, v, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=args.inner, maxcor=30, ftol=1e-15, gtol=1e-12))
        v = res.x
        m = j.metrics(v)
        ok = (m['err1'] <= ERR1 and m['err2'] <= ERR2 and m['gap'] > GAPF)
        print(f'outer {outer:3d} lam {m["lam_eff"]:.9f} e2 {m["err2"]:.9f} e1 {m["err1"]:.9f} '
              f'gap {m["gap"]:.9f} mu {mu:.0e} ok {ok} {time.time() - t0:.0f}s', flush=True)
        if ok and m['lam_eff'] < best[0]:
            best = (m['lam_eff'], v.copy())
        if m['err2'] > e2cap + 1e-8 or m['gap'] < gapcap - 1e-8:
            mu *= 10
        elif ok and m0['lam_eff'] - m['lam_eff'] < 1e-9 * m0['lam_eff']:
            break

    m1 = j.metrics(best[1])
    print('best', json.dumps(m1), flush=True)
    out = dict(f)
    w, z = j.split(best[1])
    out['W'] = w
    out['H2_I'] = z + w.sum(axis=1)
    if 'offset_W' in out:
        out['offset_W'] = z
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print('saved', args.out)


if __name__ == '__main__':
    main()

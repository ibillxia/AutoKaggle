"""Refit the DF-THC weights (w, W) to the angle vectors the circuits actually load.

The ANGLE circuits load rounded Givens vectors. lambda_eff (the first factor of the
chemistry score) and both 20 mHa accuracy gates depend on the continuous weights stored in
submission.npz, which were fitted to the *unrounded* vectors. Re-fitting them to the loaded
vectors trades accuracy margin for a lower Lambda. The circuits are untouched.

The metrics mirror dfthcsos/scoreinfra/dfthc_math.evaluate_dfthc_vectors exactly; every
candidate is re-checked with the official numpy path before it is accepted.

Usage: python chem_refit.py SUBMISSION_DIR OUT_DIR [--steps N] [--time SEC]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401

sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos.verify import (  # noqa: E402
    E_GAP_FLOOR,
    ERROR_1_THRESHOLD,
    ERROR_2_THRESHOLD,
    LAMBDA_FLOOR,
    evaluate_words,
    load_circuits,
    load_inputs,
    rejection_reason,
    resource_costs,
    score_from_metrics,
)
from dfthcsos.scoreinfra.dfthc_math import reconstruct_u_vectors  # noqa: E402


class Problem:
    """Differentiable version of evaluate_dfthc_vectors (torch, float64)."""

    def __init__(self, vectors, w, W, S, beta_1, ref, energy_upper, device='cpu'):
        import torch

        self.torch = torch
        self.dev = torch.device(device)
        to_t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), device=self.dev)
        n = int(np.asarray(ref['oei']).shape[0])
        R, B, C = np.shape(w)
        vec = np.asarray(vectors, dtype=np.float64)
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        self.N, self.R, self.B, self.C = n, R, B, C
        self.V1 = to_t(vec[:n])
        self.V2 = to_t(vec[n:].reshape(R, B, n))
        oei = np.asarray(ref['oei'], dtype=np.float64)
        tei = np.asarray(ref['tei'], dtype=np.float64)
        h1_thc = oei - 0.5 * np.einsum('prrq->pq', tei)
        self.eta, self.ecore = float(ref['electrons']), float(ref['ecore'])
        self.E_upper, self.beta_1 = float(energy_upper), float(beta_1)
        self.base = to_t(h1_thc - 0.5 * self.eta * np.asarray(S) + self.beta_1 * np.eye(n))
        # Packed upper triangle with sqrt(2) weights: same Frobenius norm over N^4, 4x cheaper.
        p, q = np.triu_indices(n)
        c = np.where(p == q, 1.0, math.sqrt(2.0))
        S = np.asarray(S, dtype=np.float64)
        eye = np.eye(n)
        G = tei[p[:, None], q[:, None], p[None, :], q[None, :]]
        G = G + 0.5 * (S[p, q][:, None] * eye[p, q][None, :] + eye[p, q][:, None] * S[p, q][None, :])
        self.Ghat = to_t(G * c[:, None] * c[None, :])
        self.pi = torch.as_tensor(p, device=self.dev)
        self.qi = torch.as_tensor(q, device=self.dev)
        self.c = to_t(c)
        self.w0, self.W0 = to_t(w), to_t(W)

    def metrics(self, w, W):
        torch = self.torch
        X = torch.einsum('rbp,rbq,rbc->rcpq', self.V2, self.V2, w)
        xhat = X[:, :, self.pi, self.qi].reshape(-1, len(self.c)) * self.c
        err2 = torch.linalg.norm(xhat.T @ xhat - self.Ghat)
        h1p = self.base - torch.einsum('rc,rcpq->pq', W, X)
        h1p = 0.5 * (h1p + h1p.T)
        ev, evec = torch.linalg.eigh(h1p)
        # The assignment is piecewise constant, so detaching it leaves the eigenvalues' gradient.
        ov = (self.V1 @ evec.detach()).abs().cpu().numpy()
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-ov)
        matched = torch.empty_like(ev)
        matched[torch.as_tensor(rows, device=self.dev)] = ev[torch.as_tensor(cols, device=self.dev)]
        rec = torch.einsum('j,jp,jq->pq', matched, self.V1, self.V1)
        err1 = torch.linalg.norm(rec - h1p)
        lam1 = ev.abs().sum()
        wB = W + w.sum(dim=1)
        Lam = lam1 + 0.25 * ((w.abs().sum(dim=1) + wB.abs()) ** 2).sum()
        esos = self.ecore - self.eta * self.beta_1 - (lam1 - ev.sum()) - 0.5 * (W ** 2).sum()
        gap = self.E_upper - esos
        leff = torch.sqrt(gap * (2 * Lam - gap))
        return {'error_1': err1, 'error_2': err2, 'Lambda': Lam, 'E_gap': gap, 'lambda_eff': leff}


def as_float(m):
    return {k: float(v.detach()) for k, v in m.items()}


def refit(prob, *, e1_max, e2_max, gap_min, steps, lr, rho, log_every, time_limit,
          tighten=0.995):
    """Penalty-method Adam on lambda_eff; keeps the best feasible iterate."""
    w = prob.w0.clone().requires_grad_(True)
    W = prob.W0.clone().requires_grad_(True)
    opt = prob.torch.optim.Adam([w, W], lr=lr)
    best, started, hist = None, time.time(), []

    def feasible(m):
        return (m['error_1'] <= e1_max and m['error_2'] <= e2_max
                and m['E_gap'] >= gap_min and m['Lambda'] > LAMBDA_FLOOR)

    for step in range(steps + 1):
        opt.zero_grad()
        m = prob.metrics(w, W)
        mf = as_float(m)
        if feasible(mf) and (best is None or mf['lambda_eff'] < best[0]['lambda_eff']):
            best = (mf, w.detach().clone(), W.detach().clone())
        t1, t2, tg = e1_max * tighten, e2_max * tighten, gap_min + 1e-4
        viol = ((prob.torch.relu(m['error_1'] - t1) / t1) ** 2
                + (prob.torch.relu(m['error_2'] - t2) / t2) ** 2
                + (prob.torch.relu(tg - m['E_gap']) / tg) ** 2)
        loss = m['lambda_eff'] + rho * viol
        if step % log_every == 0:
            hist.append({'step': step, **mf})
            b = f'{best[0]["lambda_eff"]:.4f}' if best else 'none'
            print(f'  step {step:5d} leff {mf["lambda_eff"]:.4f} Lam {mf["Lambda"]:.2f} '
                  f'gap {mf["E_gap"]:.4f} e1 {1e3*mf["error_1"]:.3f} '
                  f'e2 {1e3*mf["error_2"]:.3f} mHa  best {b}', flush=True)
        if step == steps or time.time() - started > time_limit:
            break
        loss.backward()
        opt.step()
    return best, hist


def official(P, family, words, tree, brot, angles_toffoli, w, W):
    """Official numpy metrics, resources and score for native (w, W).

    ``H2_I`` is ``w_B = W + sum_b w``; ``U`` is only read for its last dimension.
    """
    params = {**P, 'W': w, 'H2_I': W + w.sum(axis=1)}
    inp2 = load_inputs(params)
    m = evaluate_words(inp2, words, tree, brot)
    r = resource_costs(inp2, family, brot, angles_toffoli)
    return m, r, score_from_metrics(m, r), rejection_reason(m, r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('submission', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--time', type=float, default=900.0)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--angles-toffoli', type=float, default=None,
                    help='angles_toffoli of the family (default: read the source report)')
    args = ap.parse_args()

    npz = args.submission / 'submission.npz'
    P = dict(np.load(npz, allow_pickle=False))
    inp = load_inputs(npz)
    family = load_circuits(args.submission)
    brot = int(P['brot']) if 'brot' in P else 10
    words, tree = P['angle_words'], P['tree_edges']
    angles_toffoli = args.angles_toffoli
    if angles_toffoli is None:
        rep = args.submission / 'verify.json'
        if not rep.exists():
            raise SystemExit('need --angles-toffoli or a verify.json next to submission.npz')
        angles_toffoli = json.loads(rep.read_text())['resources']['angles_toffoli']

    vectors = reconstruct_u_vectors(words / (1 << (brot - 1)), tree)
    m0 = evaluate_words(inp, words, tree, brot)
    print('source metrics:', {k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in m0.items()}, flush=True)

    prob = Problem(vectors, inp.w, inp.W, inp.S, inp.beta_1, inp.ref, inp.energy_upper)
    mt = as_float(prob.metrics(prob.w0, prob.W0))
    for k in ('error_1', 'error_2', 'Lambda', 'E_gap', 'lambda_eff'):
        assert math.isclose(mt[k], m0[k], rel_tol=1e-9, abs_tol=1e-11), (k, mt[k], m0[k])
    print('torch model matches the verifier exactly', flush=True)

    lr = args.lr if args.lr is not None else 2e-3 * float(prob.w0.pow(2).mean().sqrt())
    best, _ = refit(prob,
                    e1_max=ERROR_1_THRESHOLD * 0.999, e2_max=ERROR_2_THRESHOLD * 0.999,
                    gap_min=E_GAP_FLOOR + 5e-5, steps=args.steps, lr=lr, rho=1e4,
                    log_every=max(1, args.steps // 10), time_limit=args.time)
    if best is None:
        print('no feasible iterate found')
        return
    w, W = best[1].numpy(), best[2].numpy()
    m, r, score, why = official(P, family, words, tree, brot, angles_toffoli, w, W)
    print(f'official: score {score:.6g}  leff {m["lambda_eff"]:.4f}  '
          f'e1 {1e3*m["error_1"]:.3f}  e2 {1e3*m["error_2"]:.3f} mHa  '
          f'({why or "accepted"})', flush=True)
    if not np.isfinite(score):
        print('candidate rejected; not written')
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.submission.iterdir():
        if f.name.startswith('angle_k_'):
            (args.out / f.name).write_bytes(f.read_bytes())
    np.savez(args.out / 'submission.npz',
             **{**P, 'W': w, 'H2_I': W + w.sum(axis=1)})
    print('wrote', args.out)


if __name__ == '__main__':
    main()

"""Refit the chemistry native weights Wn = H2_I - sum_b w to lower lambda_eff.

Why this exists
---------------
Our earlier LP (`chem_refit.py`, and the SCORECARD's "lambda is closed" note)
optimised the *optimizer* coordinate ``W`` with ``H2_I`` held fixed, over the
488 variables ``w`` (R x B x C) and ``w_B``.  That is a restricted slice of
the space the verifier actually prices.

The verifier's cost model is

    Lambda = sum_j |lambda_j(h1_prime)|  +  1/4 sum_rc ( |w|_1(rc) + |Wn_rc| )^2
    E_gap  = E_upper - (ecore - eta*beta - (sum|l_j| - sum l_j) - 1/2 sum Wn^2)
    lambda_eff = sqrt(E_gap * (2*Lambda - E_gap))

where ``h1_prime = M0 - sum_rc Wn_rc X_rc`` and ``Wn = H2_I - sum_b w``.

Crucially, ``Wn`` -- not ``W`` -- is what enters ``h1_prime``, so moving
``Wn`` moves the eigenvalues, both the one-body residual and the gap, while
``error_2`` stays fixed (it depends only on the decoded vectors and ``w``).
Optimising over ``Wn`` directly is therefore a strictly larger space than the
one we closed, and it is where a published competitor found headroom.

Implementation
--------------
This is a re-implementation in our own tree, not an import of the upstream
script, so that every step is auditable here.  The model mirrors the
verifier's `evaluate_words` exactly:

  * the one-body rows are the decoded ``u`` rows, matched to the eigenpairs of
    ``h1_prime`` by a maximum-|overlap| assignment (the verifier's own
    `linear_sum_assignment` rule),
  * ``error_1`` is the Frobenius residual of the matched reconstruction,
  * ``Lambda`` and ``E_gap`` follow `_exact_lambda_eff` in the verifier.

The gradient is analytic:

    d lambda_j / d Wn_rc = - v_j^T X_rc v_j

so one quasi-Newton step costs one eigendecomposition instead of 361 finite
differences.

Usage:
  python chem_native_w.py --npz IN.npz --out OUT.npz [--iters 300]
  python chem_native_w.py --npz IN.npz --out OUT.npz --verify DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as dm  # noqa: E402

ERR1_LIMIT = V.ERROR_1_THRESHOLD
GAP_FLOOR = V.E_GAP_FLOOR


class Model:
    """The verifier's chemistry metrics as a function of the native weights.

    `X` and the one-body rows are built once from the fixed words and `w`;
    only `Wn` varies.
    """

    def __init__(self, factor: dict[str, np.ndarray], brot: int = 8):
        self.inputs = V.load_inputs(factor)
        inp = self.inputs
        words = np.asarray(factor['angle_words'])
        brot = int(factor['brot']) if 'brot' in factor else int(brot)
        edges = np.asarray(factor['tree_edges'])

        u_all = dm.reconstruct_u_vectors(words / (1 << (brot - 1)), edges)
        u_all = u_all / np.linalg.norm(u_all, axis=1, keepdims=True)
        one_body, u_b = dm.split_u_vectors(u_all, inp.w)
        x = dm.reconstruct_x(u_b, inp.w)
        # the verifier symmetrises X from its lower triangle
        self.x = np.tril(x) + np.swapaxes(np.tril(x, k=-1), -1, -2)

        ref = inp.ref
        self.eta = ref['electrons']
        self.ecore = ref['ecore']
        self.e_upper = inp.energy_upper
        self.beta = inp.beta_1
        h1_thc = ref['oei'] - 0.5 * np.einsum('prrq->pq', ref['tei'])
        self.m0 = h1_thc - 0.5 * self.eta * inp.S + self.beta * np.eye(h1_thc.shape[0])
        self.one_body = np.asarray(one_body, float)
        self.w = inp.w
        self.w_l1 = np.abs(inp.w).sum(axis=1)          # (R, C)
        self.w_sum = inp.w.sum(axis=1)                 # (R, C)
        self.native0 = inp.W.copy()                   # Wn at the start point
        self.shape = self.native0.shape
        self._cache: dict[bytes, tuple] = {}

    def value_grad(self, z: np.ndarray):
        """(lambda_eff, d lambda_eff/dz, error_1, E_gap, d E_gap/dz) at z."""
        z2 = z.reshape(self.shape)
        h1p = self.m0 - np.einsum('rc,rcpq->pq', z2, self.x, optimize=True)
        ev, vec = np.linalg.eigh(h1p)
        # d lambda_j / d Wn_rc = - v_j^T X_rc v_j
        xv = np.einsum('rcpq,qj->rcpj', self.x, vec, optimize=True)
        dlam = -np.einsum('pj,rcpj->rcj', vec, xv, optimize=True)

        sg = np.sign(ev)
        lam1 = float(np.abs(ev).sum())
        d_lam1 = np.einsum('rcj,j->rc', dlam, sg)
        t = self.w_l1 + np.abs(z2 + self.w_sum)
        lam2 = float(0.25 * np.square(t).sum())
        d_lam2 = 0.5 * t * np.sign(z2 + self.w_sum)
        big = lam1 + lam2
        d_big = d_lam1 + d_lam2

        neg = float((np.abs(ev) - ev).sum())
        d_neg = np.einsum('rcj,j->rc', dlam, sg - 1.0)
        e_sos = self.ecore - self.eta * self.beta - neg - 0.5 * float(np.square(z2).sum())
        d_e_sos = -d_neg - z2
        gap = self.e_upper - e_sos
        d_gap = -d_e_sos

        rad = gap * (2 * big - gap)
        if rad <= 0:
            z0 = np.zeros_like(z2).ravel()
            return float('inf'), z0, float('inf'), gap, z0
        lam_eff = float(np.sqrt(rad))
        d_lam_eff = (d_gap * (big - gap) + gap * d_big) / lam_eff

        # one-body residual, with the verifier's maximum-|overlap| matching
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-np.abs(self.one_body @ vec))
        matched = np.empty_like(ev)
        matched[rows] = ev[cols]
        recon = np.einsum('j,jp,jq->pq', matched, self.one_body, self.one_body, optimize=True)
        err1 = float(np.linalg.norm(recon - h1p))

        return (lam_eff, d_lam_eff.ravel(), err1, gap, d_gap.ravel())

    def metrics(self, z: np.ndarray) -> dict[str, float]:
        key = z.tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        lam, _grad, err1, gap, _dgap = self.value_grad(z)
        out = {'lam_eff': lam, 'err1': err1, 'gap': gap}
        if len(self._cache) > 20000:
            self._cache.clear()
        self._cache[key] = out
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--err1-margin', type=float, default=1e-6)
    ap.add_argument('--gap-margin', type=float, default=3e-7)
    ap.add_argument('--rounds', type=int, default=3,
                    help='repeated SLSQP passes; each starts from the last point')
    ap.add_argument('--brot', type=int, default=8,
                    help='angle bits per word; only needed when the npz omits it')
    args = ap.parse_args()

    from scipy.optimize import minimize

    with np.load(args.npz, allow_pickle=False) as zf:
        factor = {k: np.array(zf[k]) for k in zf.files}

    m = Model(factor, brot=args.brot)
    z = m.native0.ravel().copy()
    m0 = m.metrics(z)
    print('start', json.dumps(m0), flush=True)

    scale = max(1e-3, float(np.abs(z).mean()))
    err_cap = ERR1_LIMIT - args.err1_margin
    gap_cap = GAP_FLOOR + args.gap_margin

    def vg(t):
        return m.value_grad(t * scale)

    def fun(t):
        return vg(t)[0]

    def jac(t):
        return vg(t)[1] * scale

    # the gap constraint carries an analytic Jacobian too: without it SLSQP
    # finite-differences 360 constraints per iteration, which costs ~13 s per
    # iteration at 35 ms per gradient.
    def gap_fun(t):
        return vg(t)[3] - gap_cap

    def gap_jac(t):
        return vg(t)[4] * scale

    t0 = time.time()
    for rnd in range(args.rounds):
        res = minimize(fun, z / scale, jac=jac, method='SLSQP',
                       constraints=[{'type': 'ineq', 'fun': gap_fun, 'jac': gap_jac}],
                       options=dict(maxiter=args.iters, ftol=1e-15))
        cand = res.x * scale
        mc = m.metrics(cand)
        print(f'round {rnd}: lam {mc["lam_eff"]:.9f} err1 {mc["err1"]:.3e} '
              f'gap {mc["gap"]:.9f} status {res.status} nit {res.nit}', flush=True)
        if mc['lam_eff'] < m.metrics(z)['lam_eff']:
            z = cand
        else:
            print('  round did not improve; stopping', flush=True)
            break

    m1 = m.metrics(z)
    print(f'end {json.dumps(m1)}  seconds {time.time() - t0:.1f}', flush=True)

    ok = (m1['lam_eff'] < m0['lam_eff']
          and m1['err1'] <= ERR1_LIMIT
          and m1['gap'] > GAP_FLOOR)
    out = dict(factor)
    if ok:
        out['H2_I'] = z.reshape(m.shape) + m.w_sum
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    report = {'start': m0, 'end': m1, 'accepted': bool(ok),
              'rel_gain': (m0['lam_eff'] - m1['lam_eff']) / m0['lam_eff']}
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

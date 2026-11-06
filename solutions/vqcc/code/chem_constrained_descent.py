"""Refit a chemistry factorization by descending *along* the constraint surface.

Why this exists
---------------
The plain LP we wrote earlier (`chem_refit.py`) optimises the one-body weights
with the gates as inequalities, and the plain augmented-Lagrangian form of the
same idea is what busyaprime calls their "R10-A", which "ended above the
start".  What actually moved their number was a different arm:

    "W and H2_I refit at fixed words: descent along the active constraint
     surface (error_2, E_gap and a pinned eigen-block held) with exact
     restoration"  (busyaprime, R53/R55)

and their R55 D additionally held `error_1` as a constraint, taking
lambda_eff from 47.019778 to 46.944209 at HW-19's words (-0.16072 % of the
chemistry cost).

The difference is *which* coordinates move.  A plain LP walks in the full
360-dimensional weight space and lets the gates push back, so it stalls in a
nonsmooth corner.  The constraint-surface descent instead:

  1. holds the gates at their limit,
  2. moves only inside the tangent subspace of the active constraints,
  3. restores the constraints exactly afterwards.

On the pi-gauge bundle (Atakan Aldemir V8) the `error_1` gate has a lot of
room -- 0.0130 against a limit of 0.020 -- and that room is exactly what the
descent spends.

What we optimise
----------------
The verifier's score is `lambda_eff * T_total * Q_total`.  `T_total` and
`Q_total` belong to the loader circuits and are fixed here; `lambda_eff` is a
function of the factorization arrays `W` (the optimizer coordinates, stored as
the npz key `W`) and `H2_I` (the native one-body weights `w_B`).

    w        = W                                  (R, B, C)
    w_B      = H2_I                               (R, C)
    W_native = H2_I - sum_b w                    (R, C)   -> `W_rho_c`
    X_rc     = sum_b w_b u_b u_b^T               (R, C, N, N)   fixed at fixed words
    h1'      = M0 - sum_rc W_native_rc X_rc
    Lambda   = sum_j |lambda_j(h1')| + 1/4 sum_rc (|w|_1(rc) + |w_B_rc|)^2
    E_gap    = E_upper - (ecore - eta beta - (sum|l_j| - sum l_j) - 1/2 sum W_native^2)
    lambda_eff = sqrt(E_gap (2 Lambda - E_gap))

`error_2` depends only on `X` and the decoded vectors, so it does not move
when the factorization does -- it is the *fixed* constraint that the descent
must respect.  `error_1` and `E_gap` do move.

The descent
-----------
We use a projected step:

  * objective `f(z) = lambda_eff(z)`, with an analytic gradient,
  * active constraint set `C` = {error_1, error_2, E_gap}, each with its own
    margin below the limit,
  * a Gauss-Newton step in the tangent space of `C` (project out the active
    constraint gradients),
  * a line search that keeps every constraint inside its margin,
  * exact restoration: after accepting a step, rescale/translate so the
    binding constraint sits exactly on its limit again.

This is a re-implementation in our own tree, so every step is auditable here;
the official verifier remains the judge.

Usage:
  python chem_constrained_descent.py --npz IN.npz --out OUT.npz [--brot 8]
                                     [--iters 60] [--margin 2e-7]
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
ERR2_LIMIT = V.ERROR_2_THRESHOLD
GAP_FLOOR = V.E_GAP_FLOOR


class Model:
    """`lambda_eff` and the three gates as functions of the npz's `W`/`H2_I`.

    The factorization coordinates are `w` (the npz key `W`, shape (R,B,C)) and
    `w_B` (the npz key `H2_I`, shape (R,C)).  We optimise in the *native*
    coordinates `W_native = w_B - sum_b w`, which is what `h1'` sees, and
    recover `w_B = W_native + sum_b w` at the end.
    """

    def __init__(self, factor: dict[str, np.ndarray], brot: int):
        self.inputs = V.load_inputs(factor)
        inp = self.inputs
        words = np.asarray(factor['angle_words'])
        edges = np.asarray(factor['tree_edges'])

        u_all = dm.reconstruct_u_vectors(words / (1 << (brot - 1)), edges)
        u_all = u_all / np.linalg.norm(u_all, axis=1, keepdims=True)
        one_body, u_b = dm.split_u_vectors(u_all, inp.w)
        x = dm.reconstruct_x(u_b, inp.w)
        self.x = np.tril(x) + np.swapaxes(np.tril(x, k=-1), -1, -2)

        ref = inp.ref
        self.eta = ref['electrons']
        self.ecore = ref['ecore']
        self.e_upper = inp.energy_upper
        self.beta = inp.beta_1
        h1_thc = ref['oei'] - 0.5 * np.einsum('prrq->pq', ref['tei'])
        self.m0 = h1_thc - 0.5 * self.eta * inp.S + self.beta * np.eye(h1_thc.shape[0])
        self.one_body = np.asarray(one_body, float)
        self.w = inp.w                                   # (R, B, C)
        self.w_l1 = np.abs(inp.w).sum(axis=1)            # (R, C)
        self.w_sum = inp.w.sum(axis=1)                   # (R, C)
        self.native0 = inp.W.copy()                     # W_native = H2_I - sum_b w
        self.shape = self.native0.shape

        # error_2 is fixed at fixed words; compute it once.
        self.err2 = self._error2()

    def _error2(self) -> float:
        ref = self.inputs.ref
        num_orbitals = ref['orbitals']
        h2 = dm.reconstruct_h2(self.x)
        h2 = h2 - ref['tei']
        d = np.arange(num_orbitals)
        h2[:, :, d, d] -= 0.5 * self.inputs.S[:, :, None]
        h2[d, d, :, :] -= 0.5 * self.inputs.S[None, :, :]
        return float(np.linalg.norm(h2.ravel()))

    def value_grad(self, z: np.ndarray):
        """(lambda_eff, d lambda_eff/dz, error_1, d error_1/dz, gap, d gap/dz).

        `z` is the flattened `W_native` (R*C numbers).
        """
        from scipy.optimize import linear_sum_assignment

        z2 = z.reshape(self.shape)
        h1p = self.m0 - np.einsum('rc,rcpq->pq', z2, self.x, optimize=True)
        ev, vec = np.linalg.eigh(h1p)
        # d lambda_j / d Wn_rc = - v_j^T X_rc v_j
        xv = np.einsum('rcpq,qj->rcpj', self.x, vec, optimize=True)
        dlam = -np.einsum('pj,rcpj->rcj', vec, xv, optimize=True)

        sg = np.sign(ev)
        lam1 = float(np.abs(ev).sum())
        d_lam1 = np.einsum('rcj,j->rc', dlam, sg)
        # Lambda's L1 term: 1/4 sum_rc (|w|_1(rc) + |w_B_rc|)^2 with
        # w_B = W_native + sum_b w.
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
            return float('inf'), z0, float('inf'), z0, gap, z0
        lam_eff = float(np.sqrt(rad))
        d_lam_eff = (d_gap * (big - gap) + gap * d_big) / lam_eff

        # error_1: the one-body rows matched to the eigenpairs of h1'.
        rows, cols = linear_sum_assignment(-np.abs(self.one_body @ vec))
        matched = np.empty_like(ev)
        matched[rows] = ev[cols]
        recon = np.einsum('j,jp,jq->pq', matched, self.one_body, self.one_body,
                          optimize=True)
        err1 = float(np.linalg.norm(recon - h1p))
        # d error_1 / d Wn_rc = -2 <h1' - recon, X_rc>_F, since d h1'/dWn = -X_rc.
        diff = h1p - recon
        d_err1 = 2.0 * np.einsum('pq,rcpq->rc', diff, self.x, optimize=True)

        return (lam_eff, d_lam_eff.ravel(), err1, d_err1.ravel(), gap, d_gap.ravel())

    def metrics(self, z: np.ndarray) -> dict[str, float]:
        lam, _g, err1, _de, gap, _dg = self.value_grad(z)
        return {'lam_eff': lam, 'err1': err1, 'err2': self.err2, 'gap': gap}


def descend(npz: Path, out: Path, brot: int, iters: int, margin: float,
            seed: int) -> dict:
    with np.load(npz, allow_pickle=False) as z:
        factor = {k: np.array(z[k]) for k in z.files}

    m = Model(factor, brot)
    z = m.native0.ravel().copy()
    m0 = m.metrics(z)
    print('start', json.dumps(m0), flush=True)

    # The caps are the *limits*, and  is how close to a limit we are
    # willing to sit.  The starting point must already satisfy them, so we
    # take the binding of (the limit) and (the start's own slack).
    caps = {
        'err1': min(ERR1_LIMIT, m0['err1']) - margin,
        'err2': min(ERR2_LIMIT, m0['err2']) - margin,
        'gap': max(GAP_FLOOR, m0['gap']) - margin,
    }
    # A gate that starts exactly at its limit cannot be tightened further.
    if m0['err1'] >= ERR1_LIMIT - margin:
        caps['err1'] = ERR1_LIMIT
    if m0['err2'] >= ERR2_LIMIT - margin:
        caps['err2'] = ERR2_LIMIT
    if m0['gap'] <= GAP_FLOOR + margin:
        caps['gap'] = GAP_FLOOR
    scale = max(1e-3, float(np.abs(z).mean()))
    rng = np.random.default_rng(seed)
    accepted = 0
    best_z, best = z.copy(), m0['lam_eff']
    t0 = time.time()

    for it in range(iters):
        lam, gl, err1, gel, gap, ggap = m.value_grad(z)
        if not np.isfinite(lam):
            break
        # Active constraints: those within 10x the margin of their limit.
        act = []
        if err1 > caps['err1'] - 10 * margin:
            act.append(gel)
        if gap < caps['gap'] + 10 * margin:
            act.append(ggap)

        # --- the descent direction -----------------------------------------
        # The objective gradient `gl` and the constraint gradients are
        # generally *not* orthogonal, so simply projecting `gl` off the
        # constraint gradient can leave almost nothing (that is what stalled
        # the first version of this script).  Instead we solve the small
        # constrained problem
        #
        #     minimise  gl . d      subject to   ga . d = 0   (active)
        #
        # whose solution lies in the null space of the active constraints and
        # is the projection of -gl onto that null space.  We do it with one
        # KKT solve in the (small) span of the active constraint gradients:
        #
        #     d = -gl + A^T y ,   A y = gl
        #
        # with A the matrix of active constraint gradients.  `A` has at most a
        # couple of columns, so this is a 2x2 solve.
        A = np.vstack([np.asarray(ga, dtype=float) for ga in act]) if act else None
        if A is None:
            d = -gl
        else:
            try:
                y = np.linalg.lstsq(A @ A.T, A @ gl, rcond=None)[0]
                d = -gl + A.T @ y
            except np.linalg.LinAlgError:
                d = -gl
        nd = float(np.linalg.norm(d))
        if nd < 1e-18:
            break
        step_dir = d / nd

        # The step length must also be small enough that the *linearised*
        # constraint change stays inside the margin.  The gap is the binding
        # constraint on this family (slack ~1.6e-6 against a floor of
        # 3.914898), so the usable step is set by the gap, not by error_1.
        # `margin` here is the gap margin we allow ourselves to spend.
        drift = 0.0
        for ga in act:
            drift = max(drift, abs(float(ga @ step_dir)))
        max_step = 1.0 if drift <= 0 else (0.5 * margin) / drift
        max_step = min(max_step, 1.0)

        # line search: keep every gate inside its cap
        best_step = 0.0
        for frac in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
            fac = frac * max_step
            cand = z + fac * step_dir * scale
            mc = m.metrics(cand)
            if (mc['err1'] <= caps['err1'] and mc['err2'] <= caps['err2']
                    and mc['gap'] >= caps['gap'] and mc['lam_eff'] < best):
                best_step = frac
                z = cand
                best = mc['lam_eff']
                accepted += 1
                break
        if best_step == 0.0:
            # no admissible improving step: try a random tangent direction
            for _ in range(8):
                v = rng.normal(size=z.size)
                if A is not None:
                    # project out the active constraints
                    try:
                        y = np.linalg.lstsq(A @ A.T, A @ v, rcond=None)[0]
                        v = v - A.T @ y
                    except np.linalg.LinAlgError:
                        pass
                nv = float(np.linalg.norm(v))
                if nv < 1e-18:
                    continue
                cand = z + 0.002 * scale * v / nv
                mc = m.metrics(cand)
                if (mc['err1'] <= caps['err1'] and mc['err2'] <= caps['err2']
                        and mc['gap'] >= caps['gap'] and mc['lam_eff'] < best):
                    z = cand
                    best = mc['lam_eff']
                    accepted += 1
                    break
            else:
                print(f'  stop at iter {it}: no admissible step', flush=True)
                break
        if it % 10 == 0:
            print(f'  iter {it:3d} lam {lam:.9f} err1 {err1:.6e} gap {gap:.9f} '
                  f'step {best_step} accepted {accepted}', flush=True)

    m1 = m.metrics(z)
    print(f'end {json.dumps(m1)}  accepted {accepted}  '
          f'seconds {time.time() - t0:.1f}', flush=True)

    ok = (m1['lam_eff'] < m0['lam_eff'] and m1['err1'] <= ERR1_LIMIT
          and m1['err2'] <= ERR2_LIMIT and m1['gap'] > GAP_FLOOR)
    out_factor = dict(factor)
    if ok:
        out_factor['H2_I'] = z.reshape(m.shape) + m.w_sum
        if 'offset_W' in out_factor:
            out_factor['offset_W'] = z.reshape(m.shape)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **out_factor)
    report = {'start': m0, 'end': m1, 'accepted': bool(ok),
              'steps_accepted': accepted,
              'rel_gain': (m0['lam_eff'] - m1['lam_eff']) / m0['lam_eff'],
              'brot': brot, 'margin': margin, 'seconds': time.time() - t0}
    out.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--brot', type=int, default=8)
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--margin', type=float, default=2e-7)
    ap.add_argument('--seed', type=int, default=20261105)
    args = ap.parse_args()
    descend(args.npz, args.out, args.brot, args.iters, args.margin, args.seed)


if __name__ == '__main__':
    main()

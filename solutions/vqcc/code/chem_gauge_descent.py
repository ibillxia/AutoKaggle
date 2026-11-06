"""Exact pairwise orthogonal-gauge descent of the DF-THC coefficient channels.

What it does
------------
For each group ``rho`` the augmented coefficient matrix is

    A_rho = [w_rho ; w_B_rho]      shape (B + 1, C)

(the ``B`` rows of the tensor ``W`` plus the row ``H2_I``).  Rotating two
channel columns ``(i, j)`` of ``A_rho`` by the same angle leaves
``A_rho A_rho^T`` unchanged, so

  * ``sum_c X_c (x) X_c`` and hence ``error_2`` is untouched,
  * ``sum_c W_c X_c`` and hence ``h1_prime`` is untouched,
  * ``sum_c W_c^2`` and hence the gap term is untouched.

Only the L1 term of the LCU one-norm moves:

    Lambda_L1 = 1/4 * sum_{rho, c} ( sum_rows |A_rho[:, c]| )^2

which is what ``lambda_eff`` prices.  For a pair of columns the rotated pair is
``(c x + s y, c y - s x)``; with the sign patterns fixed the objective is
``A + B cos 2t + C sin 2t``, so each piece between two sign changes has a
closed-form minimum.  The sign changes are the roots of ``x' = 0`` and
``y' = 0`` elements, so the global minimum on ``[-radius, radius]`` is the
best of all piece endpoints and interior stationary points.

This is a re-implementation in our own tree so that the step is auditable
here.  The Gram defect is asserted at the end: a rotation that is not exactly
orthogonal would change the physics, and the assertion catches it.

Usage:
  python chem_gauge_descent.py --npz IN.npz --out OUT.npz [--sweeps 5]
                                [--radius 0.03] [--seed 20261105]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def l1_objective(a: np.ndarray) -> float:
    return float(0.25 * np.square(np.abs(a).sum(axis=-2)).sum())


def best_angle(x: np.ndarray, y: np.ndarray, radius: float):
    """The angle minimising ``|x'|_1^2 + |y'|_1^2`` on ``[-radius, radius]``.

    The objective is piecewise smooth in the angle, with breakpoints where an
    element of ``x'`` or ``y'`` changes sign.  On each piece it is
    ``A + B cos 2t + C sin 2t``, whose minimum is at
    ``2t = atan2(-C, -B/2)``.  We evaluate every piece endpoint and every
    interior stationary point and take the best.
    """
    roots = np.concatenate([np.arctan2(-x, y), np.arctan2(y, x)])
    roots = (roots + np.pi / 2) % np.pi - np.pi / 2
    inside = roots[(roots > -radius) & (roots < radius)]
    cuts = np.unique(np.r_[-radius, 0.0, inside, radius])
    mids = (cuts[1:] + cuts[:-1]) / 2
    c, s = np.cos(mids)[:, None], np.sin(mids)[:, None]
    sx, sy = np.sign(c * x + s * y), np.sign(c * y - s * x)
    p, q = sx @ x, sx @ y          # |x'|_1 = c p + s q
    r, u = sy @ y, -(sy @ x)       # |y'|_1 = c r + s u
    aa, bb, cc = p * p + r * r, q * q + u * u, p * q + r * u
    stationary = 0.5 * np.arctan2(-cc, -(aa - bb) / 2)
    candidates = [cuts]
    for shift in (-np.pi, 0.0, np.pi):
        pts = stationary + shift
        keep = (pts > cuts[:-1]) & (pts < cuts[1:])
        candidates.append(pts[keep])
    theta = np.concatenate(candidates)
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    values = (np.square(np.abs(c * x + s * y).sum(1))
              + np.square(np.abs(c * y - s * x).sum(1)))
    k = int(np.argmin(values))
    return float(theta[k]), float(values[k])


def descend(source: Path, output: Path, sweeps: int, radius: float, seed: int):
    with np.load(source, allow_pickle=False) as z:
        data = {k: np.array(z[k]) for k in z.files}
    a0 = np.concatenate([data['W'], data['H2_I'][:, None, :]], axis=1)   # (R, B+1, C)
    a = a0.copy()
    groups, _, channels = a.shape
    gram0 = a0 @ a0.transpose(0, 2, 1)
    rng = np.random.default_rng(seed)
    pairs = [(g, i, j) for g in range(groups)
             for i in range(channels) for j in range(i + 1, channels)]
    history = []
    t0 = time.time()
    for sweep in range(sweeps):
        accepted = 0
        for index in rng.permutation(len(pairs)):
            g, i, j = pairs[index]
            x, y = a[g, :, i].copy(), a[g, :, j].copy()
            before = np.abs(x).sum() ** 2 + np.abs(y).sum() ** 2
            theta, value = best_angle(x, y, radius)
            if value < before - 1e-13:
                c, s = np.cos(theta), np.sin(theta)
                a[g, :, i], a[g, :, j] = c * x + s * y, c * y - s * x
                accepted += 1
        record = dict(sweep=sweep, objective=l1_objective(a), accepted=accepted,
                      seconds=round(time.time() - t0, 1))
        history.append(record)
        print(json.dumps(record), flush=True)
        if accepted == 0:
            break
    defect = float(np.max(np.abs(a @ a.transpose(0, 2, 1) - gram0)))
    if defect >= 1e-9:
        raise ArithmeticError(f'Gram defect {defect} -- rotation was not orthogonal')
    data['W'], data['H2_I'] = a[:, :-1, :], a[:, -1, :]
    if 'offset_W' in data:
        data['offset_W'] = data['H2_I'] - data['W'].sum(axis=1)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **data)
    report = dict(before=l1_objective(a0), after=l1_objective(a), gram_defect=defect,
                  seed=seed, radius=radius, history=history)
    Path(output).with_suffix('.json').write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--sweeps', type=int, default=5)
    ap.add_argument('--radius', type=float, default=0.03)
    ap.add_argument('--seed', type=int, default=20261105)
    args = ap.parse_args()
    rep = descend(args.npz, args.out, args.sweeps, args.radius, args.seed)
    print(json.dumps({k: v for k, v in rep.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()

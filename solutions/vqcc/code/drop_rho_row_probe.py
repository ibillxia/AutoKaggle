"""Drop whole (rho) rows -- the unit that actually shrinks R*B and T_angles.

The earlier probe zeroed individual (rho,c) entries, which leaves R unchanged.
Here whole rho rows go, so the ANGLE table loses B=60 rows per dropped rho and
T_angles falls by 2*K*B per row (K = ceil(75/3) = 25 circuits-worth of batches).
Dropping a row also removes its 30 copies from the LCU sum, so error_2 moves.

Usage: python drop_rho_row_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as M  # noqa: E402


def main() -> None:
    npz = env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz'
    z = np.load(npz)
    inp = V.load_inputs(npz)
    n = 76
    di = np.arange(n)

    X = M.reconstruct_x(z['U'], inp.w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)

    def stats(xt: np.ndarray) -> tuple[float, float]:
        rec = np.einsum('rcpq,rcst->pqst', xt, xt, optimize=True)
        d = rec - inp.ref['tei']
        d[:, :, di, di] -= 0.5 * inp.S[:, :, None]
        d[di, di, :, :] -= 0.5 * inp.S[None, :, :]
        return 1e3 * float(np.linalg.norm(d.ravel())), float((xt ** 2).sum())

    full_e, full_mass = stats(X)
    mass = np.einsum('rcpq,rcpq->r', X, X)
    order = np.argsort(mass)
    print(f'full: error_2 {full_e:.4f} mHa, sum ||X||^2 {full_mass:.4f}, R=65')
    print()
    print(f"{'drop rows':>9} {'R':>3} {'error_2':>10} {'mass kept':>10} "
          f"{'T_ang x':>8} {'passes':>7}")
    for drop in (0, 1, 2, 3, 5, 8, 12, 20):
        keep = np.ones(65, bool)
        keep[order[:drop]] = False
        xt = X.copy()
        xt[~keep] = 0.0
        e, m = stats(xt)
        rows = int(keep.sum())
        # T_angles = 2*K*(N + R*B); dropping d rows moves A = N + R*B by -d*B
        a_now = 76 + rows * 60
        a_old = 76 + 65 * 60
        tang_ratio = a_now / a_old
        print(f'{drop:>9} {rows:>3} {e:>10.4f} {100 * m / full_mass:>9.3f}% '
              f'{tang_ratio:>8.4f} {("yes" if e <= 20 else "NO"):>7}', flush=True)


if __name__ == '__main__':
    main()

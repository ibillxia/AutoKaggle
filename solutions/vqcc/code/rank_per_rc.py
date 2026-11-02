"""Can the rank-59..60 structure be exploited per-entry rather than globally?

rank_probe.py truncated EVERY X_rc to the same k, which fails at 56. But ranks
vary (57..60), and mass varies by ~100x across (rho,c). A global-k test throws
away the useful entries' tails to save the useless ones' too. This asks the
question a real cost model would: how much Frobenius mass sits in the tail
BEYOND each entry's own optimal truncation, i.e. is the failure of global-k an
averaging artefact or is every entry genuinely near-full-rank?

Usage: python rank_per_rc.py
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
    X = M.reconstruct_x(z['U'], inp.w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)

    R, C = X.shape[0], X.shape[1]
    ranks = np.empty((R, C), int)
    # effective rank at 1e-3 relative singular-value threshold
    eff = np.empty((R, C), int)
    s_all = np.empty((R, C, n))
    for r in range(R):
        for c in range(C):
            s = np.linalg.svd(X[r, c], compute_uv=False)
            s_all[r, c] = s
            ranks[r, c] = int((s > 1e-12).sum())
            eff[r, c] = int((s > 1e-3 * s[0]).sum()) if s[0] > 0 else 0

    print(f'R={R} C={C}   exact rank: min {ranks.min()} max {ranks.max()} '
          f'median {int(np.median(ranks))}')
    print(f'1e-3 relative rank: min {eff.min()} max {eff.max()} '
          f'median {int(np.median(eff))}')
    # how much mass lies beyond k per entry, expressed as total over all entries
    tot = float((s_all ** 2).sum())
    print()
    print(f"{'k':>4} {'kept':>9} {'rank<=k entries':>16}")
    for k in (30, 40, 50, 56, 57, 58, 59, 60):
        kept = float((s_all[:, :, :k] ** 2).sum()) / tot
        cnt = int((ranks <= k).sum())
        print(f'{k:>4} {100 * kept:>8.4f}% {cnt:>13} / {R * C}')
    print()
    print('Interpretation: if kept(k=56) is ~100% then global-k=56 only fails')
    print('because a few entries carry a large tail, not because mass is spread.')


if __name__ == '__main__':
    main()

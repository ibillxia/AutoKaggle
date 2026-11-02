"""Do our ANGLE tables have the algebraic structure the cheap product QROMs exploit?

`angle_words` is (num_addresses, num_angles) integers, so the loaded function is one
boolean per (angle, bit) pair: 75 angles x brot bits. Over the (rho, b) block of the
address space each such boolean is what a QROM must realise, and a plain ANF product
expansion costs about one Toffoli per monomial of degree >= 2.

Reports, on a power-of-two-aligned sub-block with no padding, the degree>=2 monomial
count per output boolean, the total, and the count of monomials shared by all of them
(exactly what a cached/product construction can amortise).

Usage: python anf_probe.py [SUBMISSION_NPZ] [--hi 32] [--lo 32]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402


def mobius2(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.uint8).copy()
    n0, n1 = m.shape
    x = 1
    while x < n0:
        for i in range(0, n0, 2 * x):
            m[i + x:i + 2 * x, :] ^= m[i:i + x, :]
        x *= 2
    x = 1
    while x < n1:
        for j in range(0, n1, 2 * x):
            m[:, j + x:j + 2 * x] ^= m[:, j:j + x]
        x *= 2
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('npz', nargs='?', default=str(env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz'))
    ap.add_argument('--hi', type=int, default=32)
    ap.add_argument('--lo', type=int, default=32)
    a = ap.parse_args()

    z = np.load(a.npz)
    inp = V.load_inputs(a.npz)
    aw = np.asarray(z['angle_words'])
    R, B, C = inp.w.shape
    N = inp.num_orbitals
    print(f'angle_words {aw.shape} {aw.dtype}  range [{aw.min()}, {aw.max()}]')
    print(f'R={R} B={B} C={C} N={N}  address space {N + R * B}  num_angles {aw.shape[1]}')

    nbits = int(aw.max()).bit_length()
    print(f'bit width implied by the data: {nbits}')

    # (num_angles, num_addresses, nbits) booleans, then the (rho, b) sub-block.
    blk = aw[N:N + R * B, :].reshape(R, B, aw.shape[1])
    blk = blk[:a.hi, :a.lo, :]
    print(f'sub-block rho<{a.hi} b<{a.lo} -> {blk.shape[0]}x{blk.shape[1]} cells, '
          f'{blk.shape[2]} angles -> {blk.shape[2] * nbits} output booleans')

    frames = [((blk[:, :, t][:, :, None] >> np.arange(nbits)) & 1)[:, :, k]
              for t in range(blk.shape[2]) for k in range(nbits)]
    anfs = [mobius2(f) for f in frames]

    totals, d2 = [], []
    for anf in anfs:
        lin = anf[0, :].sum() + anf[:, 0].sum() - anf[0, 0]
        totals.append(int(anf.sum()))
        d2.append(int(anf.sum() - lin))
    d2 = np.array(d2)
    print()
    print(f'output booleans: {len(d2)}')
    print(f'ANF monomials, all degrees: total {sum(totals)}  per boolean med {int(np.median(totals))}')
    print(f'degree >= 2 monomials: total {d2.sum()}  per boolean min {d2.min()} med {int(np.median(d2))} max {d2.max()}')
    share = np.logical_or.reduce([f.astype(bool) for f in frames])
    print(f'degree >= 2 monomials once, shared across every boolean: {mobius2(share).sum()}')
    print()
    print('Frontier reference: 262 products and ~321 Toffolis per pass for one circuit')
    print('(3 words x 8 bits = 24 output booleans), on an address space of 1366.')


if __name__ == '__main__':
    main()

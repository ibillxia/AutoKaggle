"""Marginal cost of R, B, C from the verifier's own SpinFreeData_V1.

Givens = 2*(N-1)*(2*beta-1) is independent of R, B, C, so a factorisation can
only move the outer/inner state-prep terms. This prints d(heuristic T) and
d(Q) for perturbations of the shipped (65, 60, 30) at brot=10, to show what a
successful refit would be worth BEFORE spending effort on one.

Usage: python rbc_marginal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

import numpy as np  # noqa: E402

from dfthcsos import verify as V  # noqa: E402
from dfthcsos.scoreinfra.costs_prepare import SpinFreeData_V1  # noqa: E402

N = 76
BROT = 10


def heur(R: int, B: int, C: int) -> tuple[int, int, int, int, int]:
    m = SpinFreeData_V1(R=R, B=B, C=C, N=N, chi=V.KEEP_BITS, beta=BROT)
    o, i, g = m.outer_state_prep(), m.inner_state_prep(), m.givens_rotations()
    oq, iq = m.outer_state_prep_qubit_cost(), m.inner_state_prep_qubit_cost()
    pers = int(oq['persistent']) + int(iq['persistent'])
    temp = max(int(oq['temporary']), int(iq['temporary']))
    return o, i, g, pers, pers + temp


def show(tag: str, R: int, B: int, C: int, base) -> None:
    o, i, g, pers, q = heur(R, B, C)
    t = o + i + g
    b_o, b_i, b_g, b_pers, b_q = base
    bt = b_o + b_i + b_g
    print(f'{tag:<26} R={R:<4}B={B:<4}C={C:<4} '
          f'T_h={t:<7}({t / bt - 1:+7.3%})  '
          f'Q_h={q:<4}({q / b_q - 1:+7.3%})  '
          f'[outer {o} inner {i} givens {g} pers {pers}]')


def main() -> None:
    base = heur(65, 60, 30)
    show('shipped', 65, 60, 30, base)
    print('--- R (rows) ---')
    for R in (64, 60, 56, 48, 40, 32, 16):
        show(f'drop {65 - R} rows', R, 60, 30, base)
    print('--- B (bases) ---')
    for B in (56, 48, 40, 32, 24, 16):
        show(f'B={B}', 65, B, 30, base)
    print('--- C (copies) ---')
    for C in (24, 16, 8, 4):
        show(f'C={C}', 65, 60, C, base)
    print('--- paired, keeps R*B*C-ish product ---')
    for R, B, C in ((48, 48, 30), (40, 40, 30), (32, 32, 30), (24, 24, 30),
                    (16, 16, 30), (48, 60, 24), (40, 60, 20)):
        show(f'{R}x{B}x{C}', R, B, C, base)
    print()
    print('Composite lever: score ~ lambda_eff * T * Q.  If T*Q moves by f,')
    print('composite moves by f^(1/3) when only dfthcsos changes.')


if __name__ == '__main__':
    main()

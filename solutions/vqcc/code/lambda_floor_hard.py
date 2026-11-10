"""Hard floor on Lambda / lambda_eff implied by the E_gap gate.

CORRECTED 2026-11-06.  The original version of this script used the weak
bound ``lambda_h2 >= 0.25*(||W|| - ||s||)^2`` and reported lambda_eff >=
14.46.  That is WRONG: the identity ``W = H2_I - sum_b w_b`` plus the
triangle inequality gives, pointwise for every (r,c),

    |H2_I_rc| + sum_b |w_rcb|  >=  |H2_I_rc - sum_b w_rcb| = |W_rc|,

hence ``lambda_h2 = 0.25 sum_rc (L1_rc + |H2_I_rc|)^2 >= 0.25 ||W||^2``.
The E_gap identity is  E_gap = C0 + 2p + 0.5||W||^2, so the floor
E_gap > E_GAP_FLOOR forces ||W||^2 > C - 4p with C = 1126.0947.  Combining:

    Lambda = lambda_h1 + lambda_h2 >= p + 0.25*(C - 4p) = 0.25*C = 281.5237,

INDEPENDENT of p.  Therefore lambda_eff >= sqrt(floor*(2*281.5237 - floor))
= 46.7862, matching the forum's 46.786 bound and leaving only 0.23% headroom
over the public 46.8936.  The 14.46 figure was an artifact of discarding the
dominant ||W||^2 term; do not use it.

Usage: python lambda_floor_hard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402

C_CONST = 1126.0947  # 2 * 563.047371
S_NORM = 22.8471     # ||sum_b w_b||, measured from the shipped archive


def lam_eff(E_gap: float, Lam: float) -> float:
    return float(np.sqrt(E_gap * (2.0 * Lam - E_gap)))


def main() -> None:
    floor = V.E_GAP_FLOOR
    Lam_floor = 0.25 * C_CONST
    le_floor = lam_eff(floor + 1e-9, Lam_floor)
    print(f'E_GAP_FLOOR = {floor}, C = {C_CONST}')
    print('Identity: W = H2_I - sum_b w_b, so pointwise')
    print('  |H2_I| + sum_b|w_b| >= |W|  =>  lambda_h2 >= 0.25||W||^2')
    print(f'E_gap > floor  =>  ||W||^2 > {C_CONST:.4f} - 4p')
    print(f'Lambda >= p + 0.25*(C - 4p) = 0.25*C = {Lam_floor:.4f}  '
          f'(INDEPENDENT of p)')
    print(f'HARD FLOOR: lambda_eff >= {le_floor:.4f}')
    print(f'  public lambda_eff 46.8936 -> remaining headroom '
          f'{100 * (46.8936 / le_floor - 1):.4f}%')
    print('  (The old 14.46 result came from the weak bound')
    print('   lambda_h2 >= 0.25*(||W||-||s||)^2, which drops 0.25||W||^2.)')


if __name__ == '__main__':
    main()

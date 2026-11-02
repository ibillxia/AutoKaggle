"""Cost landscape of (R,B,C) around the v9 chemistry shape (10,129,46) at brot=8.

Uses the verifier's own SpinFreeData_V1 for the heuristic part. The ANGLE part is
held at the v9 measurement (15 J5 circuits, 9,624 Toffolis) because its size follows
from the number of angle rows, which only an actual fit produces.

Usage: python cost_scan_v9.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402
from dfthcsos.scoreinfra.costs_prepare import SpinFreeData_V1  # noqa: E402

N, BROT = 76, 8
ANGLES_T = 9624          # v9 measurement at (10,129,46), J=5
LAM = 47.55333620343767  # v9 lambda_eff
BASE_COST = 3355599588.3155365


def shape(R: int, B: int, C: int):
    m = SpinFreeData_V1(R=R, B=B, C=C, N=N, chi=V.KEEP_BITS, beta=BROT)
    o, i, g = m.outer_state_prep(), m.inner_state_prep(), m.givens_rotations()
    oq, iq = m.outer_state_prep_qubit_cost(), m.inner_state_prep_qubit_cost()
    pers = int(oq['persistent']) + int(iq['persistent'])
    temp = max(int(oq['temporary']), int(iq['temporary']))
    return o + i + g, pers + temp, pers, temp


def main() -> None:
    base_t, base_q, _, _ = shape(10, 129, 46)
    base_cost = LAM * (ANGLES_T + base_t) * base_q
    print(f'base (10,129,46): T_h={base_t} Q={base_q} '
          f'cost={base_cost:.4e} ratio={BASE_COST / base_cost:.4f}')
    print(f'{"shape":<16}{"T_h":>7}{"Q":>5}{"cost":>12}{"ratio":>8}{"vs base":>9}')
    rows = []
    for R in (6, 8, 9, 10, 11, 12):
        for B in (96, 112, 129, 144, 160):
            for C in (30, 36, 40, 46, 52):
                if R * C < 290:      # Eckart-Young floor for 20 mHa
                    continue
                t, q, _, _ = shape(R, B, C)
                cost = LAM * (ANGLES_T + t) * q
                rows.append((cost, R, B, C, t, q))
    rows.sort()
    for cost, R, B, C, t, q in rows[:18]:
        print(f'({R},{B},{C}){"":<8}{t:>7}{q:>5}{cost:>12.4e}'
              f'{BASE_COST / cost:>8.3f}{base_cost / cost - 1:>+9.2%}')
    print(f'... worst of {len(rows)}:')
    for cost, R, B, C, t, q in rows[-3:]:
        print(f'({R},{B},{C}){"":<8}{t:>7}{q:>5}{cost:>12.4e}'
              f'{BASE_COST / cost:>8.3f}{base_cost / cost - 1:>+9.2%}')


if __name__ == '__main__':
    main()

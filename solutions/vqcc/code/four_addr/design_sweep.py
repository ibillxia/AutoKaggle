"""Analytic model of the four-address (banked QROM) family, calibrated on a build.

The construction folds s selection bits into 2**s - 1 difference tables and
loads all of them with one truncated-ANF pass over H = A / 2**s high addresses.
Everything is bounded by the 128-qubit cap on one ANGLE circuit:

    sel_bits + J*brot + bank_width + scratch <= 128

Calibration: the s=2, J=3 build (port_four_address.py) measured 1129 Toffolis
at H=994 with cleanup 52 and a 9*J*(2**s-1)=81 CCX row, so

    per_circuit ~= H + 9*J*(2**s-1) + cleanup(n)

and cleanup is the split-ANF erasure cost (2**low - low - 1) + (2**high - high - 1)
with low = n//2.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos.scoreinfra.costs_prepare import SpinFreeData_V1  # noqa: E402

N, NA = 76, 75
BROT = 10
SCRATCH = 5
REF_DFTHC = 3355599588.3155365


def cleanup(n):
    low = n // 2
    high = n - low
    return ((1 << low) - low - 1) + ((1 << high) - high - 1)


def heuristic(R, B, C):
    m = SpinFreeData_V1(R=R, B=B, C=C, N=N, chi=10, beta=BROT)
    oq = m.outer_state_prep_qubit_cost()
    iq = m.inner_state_prep_qubit_cost()
    pers = oq['persistent'] + iq['persistent']
    temp = max(oq['temporary'], iq['temporary'])
    return (m.outer_state_prep() + m.inner_state_prep() + m.givens_rotations(),
            pers, temp)


def evaluate(R, B, C, s, J, sequential, lam=47.85036566452195):
    """Chemistry cost of one design point, or None if it breaks the cap.

    `sequential` is the alternative to the notebook's simultaneous bank: load one
    difference table, XOR it under its selection bit, erase it, repeat. It needs
    only 9*J bank wires instead of 9*J*(2**s-1), but it pays the H-address
    truncated-ANF traversal (2**s-1) times instead of sharing it, because each
    table needs its own walk. Carrying a bank across circuits is not available
    (every circuit must return its workspace to |0>).
    """
    A = N + R * B
    sel = (A - 1).bit_length()
    H = A / (2 ** s)
    if H != int(H):
        return None
    H = int(H)
    n = (H - 1).bit_length()
    banks = 1 if sequential else (2 ** s - 1)
    bank_w = 9 * J * banks
    qubits = sel + J * BROT + bank_w + SCRATCH
    if qubits > 128:
        return None
    K = math.ceil(NA / J)
    if sequential:
        per_circ = (2 ** s - 1) * (H + 9 * J + cleanup(n))
    else:
        per_circ = H + 9 * J * (2 ** s - 1) + cleanup(n)
    T_ang = 2 * K * per_circ
    T_heur, pers, temp = heuristic(R, B, C)
    T = T_ang + T_heur
    ang_temp = qubits - sel - J * BROT
    Q = pers + J * BROT + max(temp, ang_temp)
    cost = lam * T * Q
    return {'s': s, 'J': J, 'sequential': sequential, 'K': K, 'H': H,
            'circuit_qubits': qubits, 'per_circuit': round(per_circ, 1),
            'T_angles': round(T_ang), 'T_total': round(T), 'Q_total': round(Q),
            'chemistry_cost': f'{cost:.4e}',
            'chemistry_ratio': round(REF_DFTHC / cost, 3),
            'cost_value': cost}


def main():
    rows = []
    for sequential in (False, True):
        for s in (1, 2, 3):
            for J in range(1, 13):
                r = evaluate(65, 60, 30, s, J, sequential)
                if r:
                    rows.append(r)

    rows.sort(key=lambda r: r['cost_value'])
    print(f'{"seq":>4} {"s":>2} {"J":>3} {"K":>3} {"H":>6} {"qc":>4} '
          f'{"percirc":>8} {"T_ang":>8} {"T_tot":>7} {"Q":>4} {"chem":>7}')
    for r in rows[:16]:
        print(f"{'Y' if r['sequential'] else 'n':>4} {r['s']:>2} {r['J']:>3} "
              f"{r['K']:>3} {r['H']:>6} {r['circuit_qubits']:>4} "
              f"{r['per_circuit']:>8} {r['T_angles']:>8} {r['T_total']:>7} "
              f"{r['Q_total']:>4} {r['chemistry_ratio']:>7}")

    base = 5.5531
    print()
    for r in rows[:6]:
        comp = (12.1512 * 10.6326 * r['chemistry_ratio']) ** (1 / 3)
        print(f"  s={r['s']} J={r['J']} seq={r['sequential']}: "
              f"chem {r['chemistry_ratio']:.3f}x  composite {comp:.4f}")

    out = env.SCRATCH / 'gauge' / 'design_sweep.json'
    out.write_text(json.dumps({'rows': rows, 'note': 'analytic; sequential rows are '
                               'an untested hypothesis'}, indent=1, default=str))
    print('\nwrote', out)


if __name__ == '__main__':
    main()

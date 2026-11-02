"""What the dfthcsos cost model would pay for each (B, brot) at the current lambda_eff.

T_angles is calibrated on the measured four-address build: 25 circuits,
1,129 CCX each, i.e. per-circuit ~ H + 9*J*(2**s - 1) with H = (N + R*B)/2**s
at s=2, J=3. Q_total = persistent + J*brot + max(temp, angles_temp).
The accuracy gates are NOT applied here; this is the model-only ceiling.

Usage: python chem_ceiling.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos.scoreinfra.costs_prepare import SpinFreeData_V1  # noqa: E402
from dfthcsos import verify as V  # noqa: E402

REF = 3355599588.3155365
LAMBDA = 47.85036566452195
N_ORB = 76


def per_circuit(B: int, R: int = 65, s: int = 2, J: int = 3) -> int:
    """Measured-calibrated Toffolis of one four-address ANGLE circuit."""
    H = (N_ORB + R * B) // 2 ** s
    return H + 9 * J * (2 ** s - 1)


def evaluate(R: int, B: int, C: int, brot: int, s_hi: int = 2) -> dict:
    m = SpinFreeData_V1(R=R, B=B, C=C, N=N_ORB, chi=V.KEEP_BITS, beta=brot,
                        optimize_qubits=True)
    oq, iq = m.outer_state_prep_qubit_cost(), m.inner_state_prep_qubit_cost()
    persist = oq['persistent'] + iq['persistent']
    temp = max(oq['temporary'], iq['temporary'])
    t_heur = m.outer_state_prep() + m.inner_state_prep() + m.givens_rotations()

    J = 3
    K = math.ceil(75 / J)
    t_angles = 2 * K * per_circuit(B, R, s_hi)
    t_total = t_angles + t_heur

    # ANGLE circuit width under the 128-qubit cap: sel + J*brot + 9J(2^s-1) + scratch
    angles_q = 12 + J * brot + 9 * J * (2 ** s_hi - 1) + 5
    if angles_q > 128:
        return {'feasible': False, 'angles_q': angles_q}

    q_total = persist + J * brot + temp
    cost = LAMBDA * t_total * q_total
    return {'feasible': True, 'B': B, 'brot': brot, 'J': J, 'K': K,
            'T_angles': t_angles, 'T_heur': t_heur, 'T': t_total,
            'Q': q_total, 'angles_q': angles_q,
            'cost': cost, 'ratio': REF / cost}


def main() -> None:
    rows = []
    for B in (60, 48, 40, 32, 24, 16, 12):
        for brot in (10, 9, 8, 7, 6, 5):
            r = evaluate(65, B, 30, brot)
            if r['feasible']:
                rows.append(r)
    rows.sort(key=lambda r: -r['ratio'])
    print(f"{'B':>3}{'brot':>5}{'T_angles':>9}{'T_heur':>7}{'T':>8}{'Q':>6}"
          f"{'angQ':>6}{'chem':>7}")
    for r in rows[:16]:
        print(f"{r['B']:>3}{r['brot']:>5}{r['T_angles']:>9,}{r['T_heur']:>7,}"
              f"{r['T']:>8,}{r['Q']:>6}{r['angles_q']:>6}{r['ratio']:>7.3f}")
    print()
    print('current build (B=60, brot=10, measured): 65,244 T, 661 Q -> 1.626x')
    best = rows[0]
    print(f"model ceiling here: B={best['B']} brot={best['brot']} "
          f"-> {best['ratio']:.3f}x  (accuracy gates NOT applied)")


if __name__ == '__main__':
    main()

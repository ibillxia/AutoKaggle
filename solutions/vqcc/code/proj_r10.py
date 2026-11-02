"""What the four-address chemistry would cost at the frontier shape, from our own model.

Calibration (measured on the shipped build, `design_sweep.py`):

    per_circuit = H + 9*J*(2^s - 1) + cleanup,   H = A / 2^s,   A = N + R*B
    A=3976, s=2 -> H=994, bank=81, cleanup=52  ->  1127   (official count: 1129)

Quotient: sel + J*brot + bank + scratch <= 128 (QUBIT_CAP).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers' / 'dfthcsos' / 'scoreinfra'))

from dfthcsos.scoreinfra.costs_prepare import SpinFreeData_V1 as S  # noqa: E402

N, KEEP, CAP = 76, 10, 128
BASE = 3355599588.3155365
OURS = dict(inv=13.157304419573942, dqi=14.226240291353763)


def cleanup(h):
    """Split-ANF erasure cost at the power-of-two pad of h."""
    lo = max(1, (int(h).bit_length() - 1) // 2)
    return (2 ** lo - lo - 1) + (2 ** (int(h).bit_length() - lo) - (int(h).bit_length() - lo) - 1)


def project(R, B, C, brot, J, s, lam):
    A = N + R * B
    H = A / 2 ** s
    bank = 9 * J * (2 ** s - 1)
    per = H + bank + cleanup(H)
    K = -(-(N - 1) // J)
    sel = (A - 1).bit_length()
    scratch = 5
    ang_q = sel + J * brot + bank + scratch
    if ang_q > CAP:
        return None
    m = S(R=R, B=B, C=C, N=N, chi=KEEP, beta=brot)
    ht = int(m.outer_state_prep()) + int(m.inner_state_prep()) + int(m.givens_rotations())
    oq, iq = m.outer_state_prep_qubit_cost(), m.inner_state_prep_qubit_cost()
    hpers = int(oq['persistent']) + int(iq['persistent'])
    htemp = max(int(oq['temporary']), int(iq['temporary']))
    ang_temp = ang_q - sel - J * brot
    T = ht + 2 * K * per
    Q = (hpers + J * brot) + max(htemp, ang_temp)
    cost = lam * T * Q
    return dict(A=A, H=H, per=per, K=K, T=T, Q=Q, ang_q=ang_q, cost=cost, ratio=BASE / cost)


LAM = 47.6
print(f'{"shape":22}{"brot":>5}{"J":>3}{"s":>3}{"A":>6}{"per":>6}{"K":>4}{"T":>8}{"Q":>5}{"ratio":>7}{"composite":>11}')
for tag, R, B, C, brot, J, s in [
        ('shipped 65/60/30', 65, 60, 30, 10, 3, 2),
        ('frontier 10/129/46', 10, 129, 46, 8, 3, 2),
        ('frontier 10/129/46', 10, 129, 46, 10, 3, 2),
        ('frontier 10/129/46 s3', 10, 129, 46, 8, 3, 3),
        ('frontier 10/129/60', 10, 129, 60, 8, 3, 2)]:
    p = project(R, B, C, brot, J, s, LAM)
    if p is None:
        print(f'{tag:22}{brot:>5}{J:>3}{s:>3}   does not fit the 128-qubit cap')
        continue
    comp = (OURS['inv'] * OURS['dqi'] * p['ratio']) ** (1 / 3)
    print(f'{tag:22}{brot:>5}{J:>3}{s:>3}{p["A"]:>6}{p["per"]:>6.0f}{p["K"]:>4}'
          f'{p["T"]:>8.0f}{p["Q"]:>5}{p["ratio"]:>7.2f}x{comp:>11.2f}')

print('\nReference: shipped cost 2.0636e9, ratio 1.626x, composite 6.7267')
print('Frontier (their 321/pass ANGLE, 15 circuits): ratio 9.45x, composite 12.09')

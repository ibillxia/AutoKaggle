"""Heuristic (state-prep + Givens) cost across (R,B,C,brot) shapes.

Pure formula from the official dfthcsos model -- no circuits, no fitting. Prints the
heuristic T and Q, then the total cost implied if the ANGLE family were as cheap per
circuit as the reported public frontier (~321-401 Toffolis/circuit).

Usage: python -I frontier_cost.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos.verify import heuristic_costs  # noqa: E402

N = 76
LAMBDA_EFF = 47.85037
SHIP = dict(R=65, B=60, C=30, brot=10, ang_circ=25, ang_per=50202 / 25)
PUB = dict(R=10, B=129, C=46, brot=8, ang_circ=15, ang_per=330)

def show(tag, R, B, C, brot, circ, per):
    h = heuristic_costs(N, R, B, C, brot)
    t_ang = circ * per
    T = h['toffoli'] + t_ang
    Q = h['qubits'] + circ and 0
    # Q: persistent = heur.persistent + J*brot ; J = circ? no: J = batch size
    return h, t_ang

print(f'{"shape":24} {"outer":>8} {"inner":>8} {"givens":>8} {"heurT":>8} {"heurQ":>6} {"pers":>5} {"temp":>5}')
rows = [('SHIPPED 65/60/30 b10', 65, 60, 30, 10),
        ('public 10/129/60 b8', 10, 129, 60, 8),
        ('public 10/129/46 b8', 10, 129, 46, 8),
        ('alt    10/129/46 b10', 10, 129, 46, 10),
        ('alt    20/129/46 b8', 20, 129, 46, 8),
        ('alt    10/255/46 b8', 10, 255, 46, 8)]
for tag, R, B, C, brot in rows:
    h = heuristic_costs(N, R, B, C, brot)
    print(f'{tag:24} {h["outer_state_prep_toffoli"]:>8} {h["inner_state_prep_toffoli"]:>8} '
          f'{h["givens_rotations_toffoli"]:>8} {h["toffoli"]:>8} {h["qubits"]:>6} '
          f'{h["persistent_qubits"]:>5} {h["temporary_qubits"]:>5}')

print()
print('ANGLE sensitivity (J=3 -> 25 circuits, J=5 -> 15 circuits):')
for tag, circ, per, R, B, C, brot in [
        ('shipped', 25, SHIP['ang_per'], 65, 60, 30, 10),
        ('frontier shape + our ANGLE', 25, SHIP['ang_per'], 10, 129, 46, 8),
        ('frontier shape + J5 + 401/circ', 15, 401, 10, 129, 46, 8),
        ('frontier shape + J5 + 330/circ', 15, 330, 10, 129, 46, 8)]:
    h = heuristic_costs(N, R, B, C, brot)
    T = h['toffoli'] + circ * per
    # angles persistent = J*brot, J = 75/circ
    J = 75 // circ
    ang_pers = J * brot
    ang_temp = 0  # unknown; ignore (max() usually driven by heuristic temp)
    pers = h['persistent_qubits'] + ang_pers
    temp = h['temporary_qubits']
    Q = pers + temp
    cost = LAMBDA_EFF * T * Q
    print(f'{tag:32} T={T:>7} Q={Q:>4} (pers {pers}, temp {temp})  cost={cost:>12.4e}  ratio={3355599588.3155365/cost:>6.3f}x')

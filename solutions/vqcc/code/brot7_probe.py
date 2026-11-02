"""Measure the dfthcsos accuracy gates as a function of the angle-word width.

The verifier's own `evaluate_words` scores any word table, so the quantization
question can be answered without building a single circuit. This checks the
two things that decide whether the brot<=7 cost-model cliff is reachable:
the accuracy gates, and the score the official model would give.

Usage: python brot7_probe.py [SUBMISSION_NPZ]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402
from dfthcsos.scoreinfra.costs_prepare import SpinFreeData_V1  # noqa: E402


def official_qubits(brot: int, R: int, B: int, C: int, N: int) -> dict:
    """The verifier's Q_total at this brot, with the ANGLE registers priced."""
    s = SpinFreeData_V1(R=R, B=B, C=C, N=N, chi=V.KEEP_BITS, beta=brot,
                        optimize_qubits=True)
    oq, iq = s.outer_state_prep_qubit_cost(), s.inner_state_prep_qubit_cost()
    persist = oq['persistent'] + iq['persistent']
    temp = max(oq['temporary'], iq['temporary'])
    return {'persist': persist, 'temp': temp, 'qubits_no_angles': persist + temp,
            'T_heur': s.outer_state_prep() + s.inner_state_prep() + s.givens_rotations()}


def main() -> None:
    npz = Path(sys.argv[1]) if len(sys.argv) > 1 else env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz'
    z = np.load(npz)
    words = z['angle_words']
    edges = z['tree_edges']
    print('angle_words', words.shape, words.dtype, 'max', words.max(), 'tree', edges.shape)

    inputs = V.load_inputs(npz)
    R, B, C = z['W'].shape
    N = int(inputs.num_orbitals)
    print(f'R={R} B={B} C={C} N={N} num_vectors={inputs.num_vectors}')

    rows = []
    for brot in (10, 9, 8, 7, 6):
        # Requantize: the stored words are 10-bit. Round-to-nearest on the
        # angular value, which is what `theta/pi = word / 2**(brot-1)` implies.
        shift = 10 - brot
        if shift >= 0:
            w = (words + (1 << (shift - 1))) >> shift if shift else words.copy()
            w = np.clip(w, 0, (1 << brot) - 1)
        else:
            w = words << (-shift)
        m = V.evaluate_words(inputs, w, edges, brot)
        q = official_qubits(brot, R, B, C, N)
        ok = V.passes_accuracy(m) and bool(m['gap_in_domain'])
        rows.append(dict(brot=brot, error_1_mHa=1e3 * m['error_1'],
                         error_2_mHa=1e3 * m['error_2'],
                         Lambda=m['Lambda'], E_gap=m['E_gap'],
                         lambda_eff=m['lambda_eff'], passes=bool(ok), **q))
        print(json.dumps(rows[-1]))

    print()
    print(f"{'brot':>4} {'err1':>8} {'err2':>9} {'lam_eff':>9} {'pass':>5} "
          f"{'Q_noAng':>8} {'T_heur':>7}")
    for r in rows:
        print(f"{r['brot']:>4} {r['error_1_mHa']:>8.2f} {r['error_2_mHa']:>9.2f} "
              f"{r['lambda_eff']:>9.4f} {str(r['passes']):>5} "
              f"{r['qubits_no_angles']:>8} {r['T_heur']:>7}")
    (env.SCRATCH / 'brot_probe.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()

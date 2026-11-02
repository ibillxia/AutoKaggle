"""What exactly is the binding constraint at 128 qubits?

    qubits = sel + J*brot + bank + scratch

At brot=10, s=2, J=3: 12 + 30 + 81 + 5 = 128, exactly. Every candidate improvement
either raises J (fewer circuits) or raises s (smaller H). Both need bank wires.
Two ways to buy wires back, neither used by the public notebook:

  1. Drop brot to 9 for the *word registers* only, keeping 10-bit precision in
     the bank. The gauge makes bit9 = bit8, so the word register's top bit is
     redundant with the next one - it could be a single wire instead of two.
  2. Compress the bank by GF(2) rank (measured: no saving on this data).

(1) is a real 9-wire saving at J=3, because the 10th bit of each word register is
not independent. Check it: with the gauge, is bit9 of every stored word equal to
bit8, so that one wire per word serves both?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import banked_qrom as Q  # noqa: E402
from build import signed_gauge  # noqa: E402


def main():
    z = np.load(env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz',
                allow_pickle=False)
    words = np.asarray(z['angle_words'], np.int64)
    tree = np.asarray(z['tree_edges'], np.int64)
    gauged, drift = signed_gauge(words, tree)
    print('gauge drift', drift)

    # The gauge claim: bit 9 equals bit 8 in every word.
    b9 = (gauged >> 9) & 1
    b8 = (gauged >> 8) & 1
    print(f'words with bit9 != bit8: {int((b9 != b8).sum())} of {gauged.size}')
    print(f'words with bit9 set     : {int(b9.sum())} of {gauged.size}')

    # Does the *delta* table (what the circuit actually loads) also keep that
    # redundancy? The circuit loads differences of gauged words, and XOR
    # preserves the identity only if both operands have it.
    deltas = Q.batch_deltas(gauged, 3)
    bad = sum(int((((d >> 9) & 1) != ((d >> 8) & 1)).sum()) for d in deltas)
    print(f'delta words with bit9 != bit8: {bad} of {deltas.size}')
    print()
    print('=> if 0, the top word-register bit never differs from bit 8, so a single')
    print('   wire can serve both and the word registers cost 9*J, not 10*J.')

    # Consequence for the qubit budget.
    A = words.shape[0]
    sel = (A - 1).bit_length()
    for s, J in ((2, 3), (2, 4), (3, 3), (2, 5), (3, 5)):
        now = sel + J * 10 + 9 * J * (2 ** s - 1) + 5
        opt = sel + J * 9 + 9 * J * (2 ** s - 1) + 5
        print(f's={s} J={J}: qubits {now} -> {opt} '
              f"({'fits' if opt <= 128 else 'still over'})")


if __name__ == '__main__':
    main()

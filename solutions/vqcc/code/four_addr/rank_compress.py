"""Is the difference bank compressible? Fast, vectorised check.

The bank width 9*J*(2**s-1) is what caps the family at s=2, J=3 (that point sits
exactly on the 128-qubit limit). If the difference columns were linearly
dependent over GF(2), the bank would need fewer wires and a larger s or J would
fit. Check zero columns (free wire savings) and rank, for contiguous and for
similarity-greedy row groupings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build import signed_gauge  # noqa: E402


def difference_bits(delta, s, brot=10):
    """(H, (2**s-1)*J*(brot-1)) bit matrix of the Mobius difference tables."""
    A, J = delta.shape
    G = 1 << s
    H = A // G
    v = delta.reshape(H, G, J).astype(np.int64)
    blocks = []
    for S in range(1, G):
        acc = np.zeros((H, J), np.int64)
        for a in range(G):
            if a & S == a:
                acc ^= v[:, a, :]
        bits = ((acc[:, :, None] >> np.arange(brot - 1)) & 1).astype(np.uint8)
        blocks.append(bits.reshape(H, J * (brot - 1)))
    return np.concatenate(blocks, axis=1)


def gf2_rank_cols(mat):
    """Rank over GF(2) of the columns of a uint8 bit matrix (H rows x C cols)."""
    cols = [int("".join(map(str, mat[:, c])), 2) if mat.shape[0] <= 64
            else int.from_bytes(np.packbits(mat[::-1, c]).tobytes(), 'big')
            for c in range(mat.shape[1])]
    basis = {}
    for col in cols:
        while col:
            p = col.bit_length() - 1
            if p not in basis:
                basis[p] = col
                break
            col ^= basis[p]
    return len(basis), len(cols)


def similarity_order(rows, G, rng):
    """Greedy nearest-neighbour grouping on the 9-bit word vectors."""
    A = len(rows)
    x = rows.astype(np.int64)
    used = np.zeros(A, bool)
    order = [int(rng.integers(A))]
    used[order[0]] = True
    while len(order) < A:
        while len(order) % G != 0:
            d = np.abs(x - x[order[-1]]).sum(axis=1)
            d[used] = 10 ** 9
            nxt = int(np.argmin(d))
            used[nxt] = True
            order.append(nxt)
        if len(order) < A:
            c = int(rng.choice(np.flatnonzero(~used)))
            used[c] = True
            order.append(c)
    return np.array(order)


def main():
    npz = env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz'
    z = np.load(npz, allow_pickle=False)
    gauged, _drift = signed_gauge(np.asarray(z['angle_words'], np.int64),
                                  np.asarray(z['tree_edges'], np.int64))
    A, na = gauged.shape
    rng = np.random.default_rng(20261024)
    out = []

    for s in (2, 3):
        G = 1 << s
        for J in (1, 3, 5):
            K = -(-na // J)
            padded = np.zeros((A, K * J), np.int64)
            padded[:, :na] = gauged
            batches = padded.reshape(A, -1, J).transpose(1, 0, 2)
            for name, perm in (('contiguous', np.arange(A)),
                               ('similarity', similarity_order(gauged, G, rng))):
                worst_rank = worst_cols = 0
                zero_cols = 0
                for k in range(K):
                    mat = difference_bits(batches[k][perm], s)
                    zero_cols += int((~mat.any(axis=0)).sum())
                    r, c = gf2_rank_cols(mat)
                    worst_rank = max(worst_rank, r)
                    worst_cols = max(worst_cols, c)
                row = {'s': s, 'J': J, 'grouping': name, 'K': K,
                       'raw_width': worst_cols, 'rank': worst_rank,
                       'zero_columns': zero_cols,
                       'saving_frac': round(1 - worst_rank / worst_cols, 4)}
                out.append(row)
                print(f"s={s} J={J} {name:10s} raw={worst_cols:3d} rank={worst_rank:3d} "
                      f"zero={zero_cols:3d} saving={row['saving_frac']:.2%}", flush=True)

    p = env.SCRATCH / 'gauge' / 'rank_compress.json'
    p.write_text(json.dumps(out, indent=1))
    print('wrote', p)


if __name__ == '__main__':
    main()

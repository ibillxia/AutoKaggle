"""Four-address banked QROM, implemented from scratch against the verifier.

This is the construction behind the public 6.687 notebook (Apache-2.0,
kozykappa/quantum-ai-lb-score-6-68108). Re-implemented here rather than imported,
so the circuits come from code that can be read and audited in one place, and so
the generalisation beyond s=2 can be tested.

What it does. Circuit k must map |a>|z> -> |a>|z XOR delta[k,a]>. Instead of one
QROM over all A addresses, fold the top s address bits into 2**s-1 difference
tables over the H = A/2**s remaining ("high") addresses:

    delta[h, b] = D0[h] XOR XOR_{S subset of b, S != 0} DS[h]

Load all tables for address h in one pass, then XOR each one into the word
registers under its own address-bit control, then erase the bank.

  * Loading: truncated-ANF traversal over the high address bits, which costs
    ~H monomial ANDs rather than ~H unary products.
  * Erasing: split-ANF uncompute, ~2**(n/2) Toffolis per bank.
  * The signed Givens gauge makes word bit (brot-1) equal bit (brot-2), so each
    angle needs brot-1 = 9 bank wires instead of 10.

The 128-qubit cap is what fixes the parameters: sel + J*brot + 9*J*(2**s-1) + 5.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


def batch_deltas(words, J):
    """(K, A, J) tables with delta[0]=batch[0], delta[k]=batch[k]^batch[k-1]."""
    words = np.asarray(words, np.int64)
    A, na = words.shape
    K = math.ceil(na / J)
    padded = np.zeros((A, K * J), np.int64)
    padded[:, :na] = words
    batches = padded.reshape(A, K, J).transpose(1, 0, 2)
    out = batches.copy()
    out[1:] ^= batches[:-1]
    return out


def pack_row(row, brot):
    return sum(int(v) << (j * brot) for j, v in enumerate(row))


def mobius(table):
    """Truth table <-> ANF coefficients over GF(2) (in place on a copy)."""
    r = [int(v) for v in table]
    n = (len(r) - 1).bit_length()
    for bit in range(n):
        step = 1 << bit
        for start in range(0, len(r), 2 * step):
            for off in range(step):
                r[start + step + off] ^= r[start + off]
    return r


def difference_tables(delta, s, brot=10):
    """D0 and the 2**s-1 difference tables, packed to (brot-1) bits per word.

    Returns (D0, [D_S], H) with D0[h] a J-word integer and D_S[h] a J-word
    integer whose words each fit in brot-1 bits (the gauge guarantees this).
    """
    delta = np.asarray(delta, np.int64)
    A, J = delta.shape
    G = 1 << s
    assert A % G == 0, f'A={A} is not divisible by 2**s={G}'
    H = A // G
    v = delta.reshape(H, G, J)
    base = np.array([pack_row(row, brot) for row in v[:, 0, :]], np.int64)
    tables = []
    for S in range(1, G):
        acc = np.zeros((H, J), np.int64)
        for a in range(G):
            if a & S == a:
                acc ^= v[:, a, :]
        packed = []
        for h in range(H):
            val = 0
            for j in range(J):
                w = int(acc[h, j]) & ((1 << (brot - 1)) - 1)   # gauge: bit9 == bit8
                val |= w << (j * (brot - 1))
            packed.append(val)
        tables.append(np.array(packed, np.int64))
    return base, tables, H


def xor_under(b, control, source, targets):
    """targets ^= control & source, using the first target as scratch."""
    targets = list(targets)
    if not targets:
        return
    q = targets[0]
    for t in targets[1:]:
        b.cx(q, t)
    b.ccx(control, source, q)
    for t in reversed(targets[1:]):
        b.cx(q, t)


def build_circuit(delta, s, brot=10):
    """One ANGLE_k circuit for the given (A, J) delta table."""
    import kickmix as km

    delta = np.asarray(delta, np.int64)
    A, J = delta.shape
    G = 1 << s
    bw = brot - 1          # gauge ties the top two bits of every word
    assert A % G == 0 and J * brot + bw * J * (G - 1) + 5 + (A - 1).bit_length() <= 128

    base, tables, H = difference_tables(delta, s, brot)
    n = (H - 1).bit_length()
    sel_bits = (A - 1).bit_length()

    b = km.CircuitBuilder()
    sel = b.create_quantum_register(sel_bits, 'selection')
    words = [b.create_quantum_register(brot, f'word_{j}') for j in range(J)]
    tgt = [q for w in words for q in w]
    bank = b.alloc_qubits(bw * J * (G - 1))
    # The truncated-ANF walk needs n-1 traversal carries; the notebook's 5-wire
    # figure comes from pebbling (erasing and recomputing ancestors on the fly),
    # implemented below, so 5 wires suffice here too.
    scratch = b.alloc_qubits(5)

    # a = h * 2**s + b, so b (the folded bits) is the LOW part of the address and
    # h (the high address the tables are indexed by) is the HIGH part.
    high = list(sel[s:])          # n bits indexing the H high addresses
    low = list(sel[:s])           # the s folded bits

    # Load D0 into the word registers and every D_S into the bank, in one
    # truncated-ANF pass over `high`. Rows 0..H-1 are the legal addresses; the
    # polynomial extension defines the rest and is used for the erase too.
    outputs = tgt + list(bank)
    W = J * brot
    BW = bw * J
    original = []
    for h in range(H):
        val = int(base[h])
        for S, tab in enumerate(tables):
            val |= int(tab[h]) << (W + S * BW)
        original.append(val)
    pad = [0] * ((1 << n) - H)
    coeff = mobius(original + pad)
    coeff[H:] = [0] * len(pad)
    completed = mobius(coeff)
    assert completed[:H] == original, 'truncated ANF changed a legal row'

    def load_into(mask, control):
        val = coeff[mask]
        ts = [outputs[i] for i in range(len(outputs)) if (val >> i) & 1]
        if ts:
            if control is None:
                b.x(km.array(ts))
            else:
                b.cx(control, km.array(ts))

    # Truncated-ANF walk with pebbling. The allocation rule is the notebook's:
    # a node counts toward the required depth only if it still has children, so
    # leaves (which use `direct_leaf_ands` and need no carry) do not inflate it.
    # Getting this filter wrong makes `need` one too large everywhere and makes
    # the walk pebble where it does not have to.
    @lru_cache(None)
    def children(mask, last):
        return tuple((mask | (1 << j), j) for j in range(last + 1, n)
                     if (mask | (1 << j)) < H)

    @lru_cache(None)
    def need(mask, last):
        return max([1 + need(c, j) for c, j in children(mask, last) if children(c, j)] + [0])

    free = list(scratch)

    def targets_of(mask):
        v = coeff[mask]
        return [outputs[i] for i in range(len(outputs)) if (v >> i) & 1]

    def xor_and(left, right, ts):
        """XOR left AND right into every target, sharing one Toffoli."""
        if not ts:
            return
        q = ts[0]
        for t in ts[1:]:
            b.cx(q, t)
        b.ccx(left, right, q)
        for t in reversed(ts[1:]):
            b.cx(q, t)

    def walk(mask, last, control, path):
        erased = []
        # The notebook pebbles only at bit_count == 5, the level where the five
        # carries run out; above it there are still free wires and below it the
        # remaining depth fits in the wires pebbling frees.
        if mask.bit_count() == n // 2 and need(mask, last) > len(free):
            m = need(mask, last) - len(free)
            assert 0 < m <= len(path) - 1, ('bad pebble count', m, len(path))
            erased = path[:m]
            for a, c, t in reversed(erased):
                b.del_and(a, c, t)
                free.append(t)
        load_into(mask, control)
        for child, bit in children(mask, last):
            if not mask:
                walk(child, bit, high[bit], path)
            elif not children(child, bit):
                # Leaf: no carry wire, XOR the last AND straight into the targets.
                xor_and(control, high[bit], targets_of(child))
            else:
                assert free, ('scratch exhausted', mask, bit)
                c = free.pop()
                b.init_and(control, high[bit], c)
                walk(child, bit, c, path + [(control, high[bit], c)])
                b.del_and(control, high[bit], c)
                free.append(c)
        for a, c, t in erased:
            free.remove(t)
            b.init_and(a, c, t)

    walk(0, -1, None, [])
    # Every carry is back to |0> and phase-clean here; verify that by freeing
    # only after the whole walk has uncomputed itself.

    # Fold the s low address bits in: XOR table S under the AND of its bits.
    # The AND reuses one of the walk's scratch wires, so the peak stays at
    # sel + J*brot + bank + 5 (the notebook's 128).
    both = None
    if s == 1:
        controls = [low[0]]
    else:
        both = scratch[0]
        b.init_and(low[0], low[1], both)
        controls = [low[0], low[1], both]
    for S, ctl in enumerate(controls, start=1):
        for j in range(J):
            for k in range(bw):
                ts = [tgt[j * brot + k]]
                if k == bw - 1:
                    ts.append(tgt[j * brot + brot - 1])
                xor_under(b, ctl, bank[(S - 1) * BW + j * bw + k], ts)
    if both is not None:
        b.del_and(low[0], low[1], both)
    b.free(km.array(scratch))

    # Erase the bank with a split-ANF uncompute of the completed difference tables.
    low_bits = n // 2
    bank_table = [v >> W for v in completed]
    del_anf(b, bank_table, high, list(bank), low_bits)

    b.free(km.array(bank))
    return b.finish_circuit()


def del_anf(b, table, address, target, low_bits):
    """Clear table[address] from `target` by measurement plus ANF phase correction."""
    import kickmix as km

    address = list(address)
    target = list(target)
    n = len(address)
    assert len(table) == 1 << n
    coeff = mobius(table)
    assert mobius(coeff) == list(table)

    meas = b.alloc_bits(len(target))
    b.hmr(km.array(target), meas)
    parity = b.alloc_bits(1)[0]

    low_count = 1 << low_bits
    high_bits = n - low_bits
    retained = [m for m in range(1, low_count) if m.bit_count() > 1]
    need = len(retained) + high_bits - 1
    extra = b.alloc_qubits(max(0, need - len(target)))
    scratch = target + list(extra)

    low = {1 << bit: address[bit] for bit in range(low_bits)}
    for mask, q in zip(retained, scratch):
        bit = mask.bit_length() - 1
        b.init_and(low[mask ^ (1 << bit)], address[bit], q)
        low[mask] = q
    carries = scratch[len(retained):need]

    def correct(high_mask, high_q):
        for low_mask in range(low_count):
            c = coeff[low_mask | (high_mask << low_bits)]
            if not c:
                continue
            b.bit_store(parity, False)
            while c:
                bit = (c & -c).bit_length() - 1
                b.cx(meas[bit], parity)
                c &= c - 1
            if not low_mask and not high_mask:
                b.neg_if(parity)
                continue
            b.push_condition(parity)
            if not low_mask:
                b.z(high_q)
            elif not high_mask:
                b.z(low[low_mask])
            else:
                b.cz(low[low_mask], high_q)
            b.pop_condition()

    def walk(mask, highest, control):
        correct(mask, control)
        for bit in range(highest + 1, high_bits):
            child = mask | (1 << bit)
            if not mask:
                walk(child, bit, address[low_bits + bit])
            else:
                c = carries[mask.bit_count() - 1]
                b.init_and(control, address[low_bits + bit], c)
                walk(child, bit, c)
                b.del_and(control, address[low_bits + bit], c)

    walk(0, -1, None)
    for mask in reversed(retained):
        bit = mask.bit_length() - 1
        b.del_and(low[mask ^ (1 << bit)], address[bit], low[mask])
    b.free(extra)

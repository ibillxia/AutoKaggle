# SPDX-License-Identifier: Apache-2.0
# Gu et al., deterministic field formula derivation, adapted imports.
#!/usr/bin/env python3
"""Regenerate constants.json for the dqi-reed-solomon discovered circuit.

main() reruns every search and writes constants.json.  The searches, in
file order:

  Formulas     the multiplication formulas and tower maps (KT12, KT6,
               KT3, CUBE3Q, PAIR3, LIN).  Deterministic; seconds.
  BM windows   the per-round coefficient windows (IMAXV, KMSV, KMMV).
               search.c samples the full Berlekamp-Massey recursion over
               403.2M weight-law draws (56 fixed seeds, 7.2M draws each)
               and counts, per (round, slot), how often the slot changes
               state.  Each window is the smallest one whose clipped
               slots all have mass under 4e-5 per draw.  About 2.7
               CPU-hours; the seeds run on a thread pool (3.5 min on 56
               cores).
  Psi windows  the cofactor windows of the Psi route: per CR op, the
               minimal cyclic interval covering the union support where
               the op acts, over 400k law draws (16 fixed seeds) plus
               the w = 0 anchor and every w = 1 position.  BM runs under
               the freshly derived BM windows.

The models section holds the bit-exact classical models of the circuit
that the Psi window search runs: the fused-Q BM recursion and the Psi
route (stages 5-7).  The script derives the instance from contract.json
and does not import circuit.py, so it runs without constants.json.

Formula derivation.  The field is GF(2^12), poly 4331, x = 2
primitive.  The tower is GF(2^12) = GF(64)(t) = GF(8)(s)(t), read top
down: A = x + x^64, t = x / A, nu = x^65 / A^2, u = x^65,
A' = u + u^8, s = u / A', nu' = u^9 / A'^2, v = u^9.  Tower
coordinates are E12[i + 3j + 6k] = v^i s^j t^k.  The norm descends
both quadratic levels (N = x xbar in GF(64), then M = N Nbar in GF(8)),
so the only nonlinear inverse left is M -> M^3 in GF(8), a 3-product
quadratic form; M^-1 = (M^3)^2 is a free squaring.  A full inversion
costs N (15) + M (6) + M^3 (3) + Nbar M^-1 (12) + xbar N^-1 (30) = 66
Toffoli, and every stage maps 0 to 0.  Rejected towers: GF(64)^2 alone
needs N^31 (five 1-bits, so a generic GF(64) product on top of two
quadratic forms); GF(16)^3 pays a cubic norm; plain Itoh-Tsujii needs
at least four 42-Toffoli products.

  KT12    GF(2^12) = GF(4)[x]: a 6-term product over GF(4) by CRT at
          the four GF(4) points, infinity, and the irreducible
          quadratics x^2+x+w, x^2+x+w^2, x^2+wx+1.  14 GF(4) products
          x 3 = 42; the greedy prune removes nothing.  GF(8)-CRT and
          Chudnovsky-style curves over GF(4) tie at 42; Karatsuba
          variants give 45-51.
  KT6     GF(64) = GF(4)[u], 3 terms, CRT at four points + infinity:
          15 products, the best known bilinear complexity of GF(2^6).
  KT3     the 3-term formula M3TERMS reduced modulo v's minimal
          polynomial: 6 products.
  CUBE3Q  exhaustive search over product subsets: 3 is the minimum.
  PAIR3   elimination rank decomposition of Tr3(r s^2): rank 3.
  LIN     the tower's linear maps (pure linear algebra on the
          elements above).

RUN: uv run python gen_constants.py   (writes constants.json)
"""

import collections
import ctypes
import json
import os
import pathlib
import random
import sys
import time
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np

from port_runtime import GF2n, f2_matinv, f2_transpose, parity
from port_runtime import load_contract

HERE = pathlib.Path(__file__).resolve().parent
PATH = HERE / "constants.json"

# The instance (contract.json), derived as circuit.py derives it.
CONTRACT = load_contract(__file__)
M = CONTRACT["m"]  # code length = nonzero field points
N = CONTRACT["n_synd"]  # syndrome symbols = BM rounds
B = CONTRACT["b"]  # bits per symbol
W = CONTRACT["w_max"]  # unique-decoding radius n/2 = max locator degree
LAW_CUM = CONTRACT["weight_cum_2p16"]  # the grader's 2^16 cumulative weight law
NH = W // 2 + 1  # words per Euclid half E', O' (and per cofactor)
FIELD = GF2n(B, CONTRACT["gf_poly"])
EXP, LOG = FIELD.exp, FIELD.log
gmul, ginv, gsqrt, trace = FIELD.mul, FIELD.inv, FIELD.sqrt, FIELD.trace
DWB = (N - 2).bit_length() + 1  # the signed d counter dW: two's complement
DMAX = 1 << (DWB - 1)  # two's-complement range of dW


def bitlist(m, n):
    return [i for i in range(n) if (m >> i) & 1]


def bitmask(n, pred):
    return sum(1 << i for i in range(n) if pred(i))


DWMASK = (1 << DWB) - 1  # the dW register (two's complement, DWB bits)
MMASK = 31  # m = (dW + 2) >> 2 lives in 5 bits (dW[2:7])
# The cores this process may use (its CPU affinity set, not the whole node).
NCPU = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()

# #########################################################################
# Formulas
# #########################################################################
# Every table is deterministic; the module docstring gives the derivation.
# search_kt keeps the CRT formula only if certify_kt accepts it.

# ---------------------------------------------------------------------------
# small F2 helpers
# ---------------------------------------------------------------------------


def rows_apply(rows, x):
    """bit r of the result = parity(rows[r] & x)."""
    v = 0
    for r in range(len(rows)):
        if parity(rows[r] & x):
            v |= 1 << r
    return v


def compose_rows(A, B, nb):
    """Rows of A o B (apply B, then A); B has nb rows."""
    out = []
    for r in range(len(A)):
        m = 0
        for i in bitlist(A[r], nb):
            m ^= B[i]
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# GF(2^b) arithmetic
# ---------------------------------------------------------------------------


def reduce_raw(raw):
    """Reduce a raw F2[x] product (degree <= 2B - 2) mod the field
    polynomial: x^k = EXP[k], since gamma = x.
    """
    v = 0
    for k in bitlist(raw, 2 * B - 1):
        v ^= EXP[k]
    return v


# ---------------------------------------------------------------------------
# polynomial-multiplication formulas (Karatsuba fallback, GF(8) terms)
# ---------------------------------------------------------------------------

M3TERMS = [
    (1, 1, 0b00111),
    (2, 2, 0b01110),
    (4, 4, 0b11100),
    (3, 3, 0b00010),
    (6, 6, 0b01000),
    (5, 5, 0b00100),
]


def kara(n):
    """Recursive symmetric Karatsuba formula for n-term F2 polynomial
    multiplication, [(Amask, Bmask, raw-product mask)].
    """
    if n == 1:
        return [(1, 1, 1)]
    if n == 3:
        return list(M3TERMS)
    n0 = n // 2
    n1 = n - n0
    m = max(n0, n1)
    out = []
    for A, B, C in kara(n0):
        out.append((A, B, C ^ (C << n0)))
    for A, B, C in kara(n1):
        out.append((A << n0, B << n0, (C << n0) ^ (C << (2 * n0))))
    for A, B, C in kara(m):
        Am = 0
        Bm = 0
        for i in range(m):
            if (A >> i) & 1:
                if i < n0:
                    Am ^= 1 << i
                if i < n1:
                    Am ^= 1 << (n0 + i)
            if (B >> i) & 1:
                if i < n0:
                    Bm ^= 1 << i
                if i < n1:
                    Bm ^= 1 << (n0 + i)
        out.append((Am, Bm, C << n0))
    return out


# ---------------------------------------------------------------------------
# GF(4)-tower CRT bilinear formulas (deterministic; KT / KT12 / KT6)
# ---------------------------------------------------------------------------


class F4:
    """GF(4) = {0, 1, w, w+1} inside fld, index-encoded (bit 0 = 1, bit 1 =
    w), so index addition is XOR.
    """

    def __init__(self, fld):
        gmul = fld.mul
        w3 = fld.exp[fld.order // 3]
        assert gmul(w3, w3) == (w3 ^ 1) and gmul(w3, w3 ^ 1) == 1
        self.w3 = w3
        self.E = [0, 1, w3, w3 ^ 1]
        self.I = {self.E[i]: i for i in range(4)}
        self.gmul = gmul

    def mul(self, a, bq):
        return self.I[self.gmul(self.E[a], self.E[bq])]


def f4_irreducible_quadratics(f4):
    """Monic irreducible x^2 + a x + b over GF(4), (a, b) in index order."""
    out = []
    for a in range(4):
        for bq in range(4):
            if all((f4.mul(r, r) ^ f4.mul(a, r) ^ bq) != 0 for r in range(4)):
                out.append((a, bq))
    assert len(out) == 6
    return out


def f4_lams(f4, k, nquad):
    """Return the CRT evaluation functionals for a k-term product over GF(4):
    evaluation at 0, 1, w, w^2, at infinity, then for each of the first
    nquad irreducible quadratics q the two reduction coordinates
    (x^i mod q = c0_i + c1_i x) and their sum (Karatsuba over GF(4)).
    Modulus degree = 4 + 1 + 2 nquad; the product has 2k - 1 coefficients.
    """
    assert 4 + 1 + 2 * nquad == 2 * k - 1, (k, nquad)
    lams = []
    for pt in range(4):
        lam, p = [], 1
        for _ in range(k):
            lam.append(p)
            p = f4.mul(p, pt)
        lams.append(lam)
    lams.append([0] * (k - 1) + [1])
    for a, bq in f4_irreducible_quadratics(f4)[:nquad]:
        L0, L1 = [], []
        c0, c1 = 1, 0
        for _ in range(k):
            L0.append(c0)
            L1.append(c1)
            c0, c1 = f4.mul(c1, bq), c0 ^ f4.mul(c1, a)
        lams.extend([L0, L1, [p ^ q for p, q in zip(L0, L1)]])
    return lams


def f4_digits(f4, fld, coords, gens, k):
    """GF(4)-digit extraction for an n = 2k dimensional coordinate system:
    gens[i] (i < k) are field elements whose GF(4)-span {gens[i], gens[i]
    w} is the space; coords(elt) -> n-bit coordinate vector.  Returns
    DIG[bq] = the k GF(4)-digit indices of coordinate basis vector bq.
    """
    n = 2 * k
    colsE = []
    for i in range(k):
        for j in range(2):
            e = gens[i]
            if j:
                e = fld.mul(e, f4.w3)
            colsE.append(coords(e))
    EXTR = f2_matinv(f2_transpose(colsE, n), n)
    DIG = []
    for bq in range(n):
        c = rows_apply(EXTR, 1 << bq)
        DIG.append([(c >> (2 * i)) & 3 for i in range(k)])
    return DIG


def solve_sym(n, prodlist, mulcoord):
    """Reconstruction solve for symmetric operand masks: target masks T[t]
    per product such that a*b = XOR_t parity(a&m_t) parity(b&m_t) T[t] on all
    n x n basis pairs (verified), or None.
    """
    piv = {}
    for i in range(n):
        for j in range(n):
            a = 0
            for t, mk in enumerate(prodlist):
                if ((mk >> i) & 1) and ((mk >> j) & 1):
                    a |= 1 << t
            r = mulcoord(i, j)
            while a:
                cc = a.bit_length() - 1
                if cc in piv:
                    pa, pr = piv[cc]
                    a ^= pa
                    r ^= pr
                else:
                    piv[cc] = (a, r)
                    a = 0
                    r = 0
            if r:
                return None
    T = [0] * len(prodlist)
    for cc in sorted(piv):
        a, r = piv[cc]
        acc = r
        aa = a & ~(1 << cc)
        while aa:
            c2 = aa.bit_length() - 1
            aa ^= 1 << c2
            acc ^= T[c2]
        T[cc] = acc
    for i in range(n):
        for j in range(n):
            acc = 0
            for t, mk in enumerate(prodlist):
                if ((mk >> i) & 1) and ((mk >> j) & 1):
                    acc ^= T[t]
            if acc != mulcoord(i, j):
                return None
    return T


def crt_f4_formula(n, DIG, k, lams, f4mul, mulcoord):
    """GF(4)-CRT bilinear formula on n = 2k coordinate bits: products =
    3 F2 parities per lambda, reconstruction solved by
    Gaussian elimination over all n x n basis pairs, deterministic greedy
    prune in index order.  Returns [(prodmask, targetmask)] (symmetric
    operand masks) or None.
    """
    prods = []
    for lam in lams:
        m0 = 0
        m1 = 0
        for bq in range(n):
            U = 0
            for i in range(k):
                U ^= f4mul(lam[i], DIG[bq][i])
            if U & 1:
                m0 |= 1 << bq
            if U & 2:
                m1 |= 1 << bq
        prods.extend([m0, m1, m0 ^ m1])
    assert len(prods) == 3 * len(lams)

    T = solve_sym(n, prods, mulcoord)
    if T is None:
        return None
    keep = list(range(len(prods)))
    for t in range(len(prods)):
        if t not in keep:
            continue
        trial = [prods[u] for u in keep if u != t]
        if solve_sym(n, trial, mulcoord) is not None:
            keep = [u for u in keep if u != t]
    plist = [prods[u] for u in keep]
    T = solve_sym(n, plist, mulcoord)
    if T is None:
        return None
    out = []
    for t, mk in enumerate(plist):
        if T[t] and mk:
            out.append((mk, T[t]))
    return out


def build_f4_crt_formula(fld):
    """GF(2^b) multiplication over the poly basis via GF(2^b) = GF(4)[x]:
    b/2 terms over GF(4), moduli = 4 points + infinity + (b/2 - 3)
    irreducible quadratics (b = 12: 42 products).
    """
    b = fld.n
    k = b // 2
    try:
        f4 = F4(fld)
        theta = 2
        gens = [fld.pow(theta, i) for i in range(k)]
        DIG = f4_digits(f4, fld, lambda e: e, gens, k)
        for v in range(1 << b):  # exhaustive digit round-trip
            dg = [0] * k
            for bq in bitlist(v, b):
                dg = [x ^ y for x, y in zip(dg, DIG[bq])]
            w = 0
            for i in range(k):
                w ^= fld.mul(gens[i], f4.E[dg[i]])
            assert w == v
        lams = f4_lams(f4, k, k - 3)
        return crt_f4_formula(b, DIG, k, lams, f4.mul, lambda i, j: fld.mul(1 << i, 1 << j))
    except Exception as ex:  # noqa: BLE001  (best-effort alternative: any failure falls back to kara(b))
        print("f4-crt formula construction failed:", repr(ex))
        return None


def search_kt(fld):
    """Return the GF(2^b) multiplication formula, constants.json shape
    [[Ab, Bb, Tb], ...].  Deterministic: the Karatsuba formula kara(b)
    (b = 12: 54 products), replaced by the GF(4)-tower CRT formula when
    that is smaller.
    """
    b = fld.n
    RAW = kara(b)
    # basis-exhaustive check of RAW (complete for a bilinear form)
    for i in range(b):
        for j in range(b):
            pr = 0
            for Am, Bm, Cm in RAW:
                if parity((1 << i) & Am) and parity((1 << j) & Bm):
                    pr ^= Cm
            assert reduce_raw(pr) == fld.mul(1 << i, 1 << j)
    KT = []
    for Am, Bm, Cm in RAW:
        tm = reduce_raw(Cm)
        if tm:
            KT.append([bitlist(Am, b), bitlist(Bm, b), bitlist(tm, b)])
    F4KT = build_f4_crt_formula(fld)
    if F4KT is not None:
        KT_NEW = [[bitlist(mk, b), bitlist(mk, b), bitlist(tm, b)] for (mk, tm) in F4KT]
        if certify_kt(KT_NEW, fld) and len(KT_NEW) < len(KT):
            KT = KT_NEW
    return KT


# ---------------------------------------------------------------------------
# The towers
# ---------------------------------------------------------------------------


class Tower:
    """The (2, 2, 3) tower of the module docstring.  GF(2^12) = GF(64)(t),
    GF(64) = GF(8)(s), GF(8) = F2[v]; E_list = [v^i s^j t^k] indexed
    i + 3j + 6k; the GF(64) coordinates are E6 = [v^i s^j] (i + 3j), the
    GF(8) coordinates E3 = [v^i].
    """

    def __init__(self, fld):
        self.fld = fld
        b = fld.n
        assert b == 12
        h = b // 2
        EXP, gmul, ginv, gpow = fld.exp, fld.mul, fld.inv, fld.pow
        Ael = EXP[1 << h] ^ 2  # x^(2^h) + x = Tr(x)
        t_el = gmul(2, ginv(Ael))  # t = x / A
        u_el = EXP[(1 << h) + 1]  # u = x^(2^h + 1) = N(x)
        nu_el = gmul(u_el, ginv(gmul(Ael, Ael)))  # nu = u / A^2
        assert gpow(u_el, 1 << h) == u_el and u_el != 0
        assert gpow(Ael, 1 << h) == Ael and Ael != 0
        assert gpow(nu_el, 1 << h) == nu_el
        assert gmul(t_el, t_el) ^ t_el == nu_el
        self.Ael, self.t_el, self.u_el, self.nu_el = Ael, t_el, u_el, nu_el
        self.h = h
        # sub-level GF(64) = GF(8)(s)
        Ap = gpow(u_el, 8) ^ u_el  # Tr_{6/3}(u)
        s_el = gmul(u_el, ginv(Ap))
        v_el = gpow(u_el, 9)  # N_{6/3}(u)
        nup_el = gmul(v_el, ginv(gmul(Ap, Ap)))
        assert gpow(Ap, 8) == Ap and Ap != 0
        assert gpow(v_el, 8) == v_el and gpow(v_el, 7) == 1 and v_el != 1
        assert gpow(nup_el, 8) == nup_el
        assert gmul(s_el, s_el) ^ s_el == nup_el
        self.Ap, self.s_el, self.v_el, self.nup_el = (Ap, s_el, v_el, nup_el)
        E3 = [gpow(v_el, i) for i in range(3)]
        E6 = E3 + [gmul(e, s_el) for e in E3]
        E_list = E6 + [gmul(e, t_el) for e in E6]
        self.E_sub = E6
        self.E3 = E3
        self.E_list = E_list
        self.MINVR = f2_matinv(f2_transpose(E_list, b), b)
        for v in range(1 << b):
            c = self.coords_full(v)
            w = 0
            for u in bitlist(c, b):
                w ^= E_list[u]
            assert w == v
        # projections poly -> (a0, a1) and the conjugate's constant part
        self.A0ROWS = self.MINVR[0:h]
        self.A1ROWS = self.MINVR[h:b]
        self.A01ROWS = [self.A0ROWS[r] ^ self.A1ROWS[r] for r in range(h)]
        # squaring and multiply-by-nu on subfield coords
        self.SQR = f2_transpose([self.coords_sub(gmul(e, e)) for e in self.E_sub], h)
        self.NUR = f2_transpose([self.coords_sub(gmul(nu_el, e)) for e in self.E_sub], h)
        NSR = compose_rows(self.NUR, self.SQR, h)
        # linear part of the norm N = a0 a1 + a0^2 + nu a1^2 on poly words
        self.LNROWS = []
        for r in range(h):
            m = 0
            for j in bitlist(self.SQR[r], h):
                m ^= self.A0ROWS[j]
            for j in bitlist(NSR[r], h):
                m ^= self.A1ROWS[j]
            self.LNROWS.append(m)
        # GF(8) level: coords3 = low 3 bits of the GF(64) coords of a
        # GF(8) element; RED3 = v^k for the KT3 reduction
        self.RED3 = [self.coords3(gpow(v_el, k)) for k in range(5)]
        self.SQ3R = f2_transpose([self.coords3(gmul(e, e)) for e in self.E3], 3)
        self.NU3R = f2_transpose([self.coords3(gmul(nup_el, e)) for e in self.E3], 3)
        NSR3 = compose_rows(self.NU3R, self.SQ3R, 3)
        # linear part of M = n0 n1 + n0^2 + nu' n1^2 on GF(64) coords
        self.LMROWS = [self.SQ3R[r] ^ (NSR3[r] << 3) for r in range(3)]
        self.FCUBE3 = [self.coords3(gpow(self.elt3(c), 3)) for c in range(8)]

    # -- coordinates --
    def coords_full(self, v):
        return rows_apply(self.MINVR, v)

    def coords_sub(self, v):
        c = self.coords_full(v)
        assert c < (1 << self.h), "not a subfield element"
        return c

    def coords3(self, v):
        c = self.coords_full(v)
        assert c < 8, "not a GF(8) element"
        return c

    def elt3(self, c):
        w = 0
        for i in bitlist(c, 3):
            w ^= self.E3[i]
        return w

    def lin_tables(self):
        """Return the LIN block of constants.json: the linear maps circuit.py
        reads.
        """
        return {
            "E12": list(self.E_list),
            "A0ROWS": list(self.A0ROWS),
            "A1ROWS": list(self.A1ROWS),
            "LNROWS": list(self.LNROWS),
            "SQ3R": list(self.SQ3R),
            "LMROWS": list(self.LMROWS),
        }


# ---------------------------------------------------------------------------
# subfield multiplication formulas
# ---------------------------------------------------------------------------


def search_kt6(fld, tower):
    """Return the GF(64) multiplication formula in tower coords (E6), 15
    products: GF(64) = GF(4)[u], 3 terms over GF(4), CRT at the four points
    + infinity, built directly on the 6-bit coordinate vectors.
    """
    f4 = F4(fld)
    u = tower.u_el
    gens = [fld.pow(u, i) for i in range(3)]
    DIG = f4_digits(f4, fld, tower.coords_sub, gens, 3)
    lams = f4_lams(f4, 3, 0)
    assert len(lams) == 5
    E6 = tower.E_sub
    F = crt_f4_formula(
        6, DIG, 3, lams, f4.mul, lambda i, j: tower.coords_sub(fld.mul(E6[i], E6[j]))
    )
    assert F is not None, "GF(64) CRT formula failed"
    return [[bitlist(mk, 6), bitlist(mk, 6), bitlist(tm, 6)] for (mk, tm) in F]


def search_kt3(tower):
    """Return the GF(8) multiplication formula in the v-basis, 6 products:
    M3TERMS reduced modulo the minimal polynomial of v.
    """
    KT3 = []
    for Am, Bm, Cm in M3TERMS:
        tm = 0
        for kk in range(5):
            if (Cm >> kk) & 1:
                tm ^= tower.RED3[kk]
        if tm:
            KT3.append([bitlist(Am, 3), bitlist(Bm, 3), bitlist(tm, 3)])
    return KT3


# ---------------------------------------------------------------------------
# quadratic-form formulas: exact solve and exhaustive search (CUBE3Q)
# ---------------------------------------------------------------------------


def qp_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _offvec(n, u, w):
    v = 0
    for idx, (i, j) in enumerate(qp_pairs(n)):
        if (((u >> i) & (w >> j)) ^ ((u >> j) & (w >> i))) & 1:
            v |= 1 << idx
    return v


def quad_solve(n, prods, fvals):
    """Given product pairs (u, w),
    solve for output masks + linear correction so the quadratic formula
    reproduces fvals on all 2^n elements; returns (tris, corr) or None.
    """
    pairs = qp_pairs(n)
    bas = {}
    for t_, (u, w) in enumerate(prods):
        x = _offvec(n, u, w)
        sel = 1 << t_
        while x:
            mb = x.bit_length() - 1
            if mb in bas:
                x ^= bas[mb][0]
                sel ^= bas[mb][1]
            else:
                bas[mb] = (x, sel)
                break
    tg = [0] * n
    for idx, (i, j) in enumerate(pairs):
        cdiff = fvals[(1 << i) | (1 << j)] ^ fvals[1 << i] ^ fvals[1 << j]
        for m in range(n):
            if (cdiff >> m) & 1:
                tg[m] |= 1 << idx
    outs = []
    for m in range(n):
        x = tg[m]
        sel = 0
        while x:
            mb = x.bit_length() - 1
            if mb not in bas:
                return None
            x ^= bas[mb][0]
            sel ^= bas[mb][1]
        outs.append(sel)
    tris = []
    for t_, (u, w) in enumerate(prods):
        om = 0
        for m in range(n):
            if (outs[m] >> t_) & 1:
                om |= 1 << m
        if om:
            tris.append((u, w, om))
    corr = [0] * n
    for i in range(n):
        cv = fvals[1 << i]
        for u, w, om in tris:
            if (u >> i) & 1 and (w >> i) & 1:
                cv ^= om
        for m in range(n):
            if (cv >> m) & 1:
                corr[m] |= 1 << i
    for x in range(1 << n):
        if quad_apply((tris, corr), x, n) != fvals[x]:
            return None
    return (tris, corr)


def quad_exhaustive(n, fvals):
    """EXHAUSTIVE minimum-product quadratic formula for fvals on n bits:
    candidate products are the unordered pairs u < w of nonzero masks (a
    pair (u, u) is linear and never needed); subsets are tried in
    increasing size, lexicographic order, and the first solvable one wins.
    Feasible for n <= 4; used for GF(8).
    """
    from itertools import combinations

    cands = [(u, w) for u in range(1, 1 << n) for w in range(u + 1, 1 << n)]
    for r in range(len(qp_pairs(n)) + 1):
        for sub in combinations(cands, r):
            res = quad_solve(n, list(sub), fvals)
            if res is not None:
                tris, corr = res
                return [[list(t) for t in tris], list(corr)]
    raise AssertionError("no quadratic formula (impossible)")


def search_cube3q(tower):
    """Exhaustive, deterministic: x -> x^3 in GF(8), v-basis."""
    return quad_exhaustive(3, tower.FCUBE3)


# ---------------------------------------------------------------------------
# pairing-form rank decomposition (elimination)
# ---------------------------------------------------------------------------


def pair_decomp(Brows, n):
    """B[u] = mask over v of the form on basis pair (u, v).  Returns
    [(p, qm), ...] with B(a, c) = XOR_k parity(a & p_k) & parity(c & qm_k).
    """
    Rw = list(Brows)
    out = []
    while True:
        u0 = None
        for u in range(n):
            if Rw[u]:
                u0 = u
                break
        if u0 is None:
            break
        qm = Rw[u0]
        v0 = (qm & -qm).bit_length() - 1
        p = bitmask(n, lambda u: (Rw[u] >> v0) & 1)
        for u in bitlist(p, n):
            Rw[u] ^= qm
        out.append((p, qm))
    chk = [0] * n
    for p, qm in out:
        for u in bitlist(p, n):
            chk[u] ^= qm
    assert chk == list(Brows)
    return out


def search_pair3(fld, tower):
    """Rank decomposition of B(r, s) = Tr3(r s^2) on GF(8) v-coords."""
    gmul = fld.mul
    E3 = tower.E3
    B33 = [bitmask(3, lambda j: trace(gmul(E3[i], gmul(E3[j], E3[j])), 3)) for i in range(3)]
    return [[p, qm] for (p, qm) in pair_decomp(B33, 3)]


# ---------------------------------------------------------------------------
# formula evaluation (search_kt's certify_kt, quad_solve's quad_apply)
# ---------------------------------------------------------------------------


def kt_mul_fn(KT):
    ktm = [
        (sum(1 << u for u in Ab), sum(1 << u for u in Bb), sum(1 << u for u in Tb))
        for (Ab, Bb, Tb) in KT
    ]

    def mul(a, bb):
        acc = 0
        for Am, Bm, Tm in ktm:
            if parity(a & Am) and parity(bb & Bm):
                acc ^= Tm
        return acc

    return mul


def certify_kt(KT, fld):
    """Return True iff KT multiplies exactly in GF(2^b) over the poly basis:
    checked on the b x b basis pairs, complete for a bilinear form.
    """
    kt_mul = kt_mul_fn(KT)
    return all(
        kt_mul(1 << i, 1 << j) == fld.mul(1 << i, 1 << j)
        for i in range(fld.n)
        for j in range(fld.n)
    )


def quad_apply(Q, x, n):
    tris, corr = Q
    acc = 0
    for u, w, om in tris:
        if parity(u & x) and parity(w & x):
            acc ^= om
    for m in range(n):
        if parity(corr[m] & x):
            acc ^= 1 << m
    return acc


def run_b12():
    """All formula searches (every one deterministic); returns (tables,
    timings) with tables in constants.json shape.
    """
    fld = FIELD
    tower = Tower(fld)
    tables, tm = {}, {}
    for name, search in (
        ("KT12", lambda: search_kt(fld)),
        ("KT6", lambda: search_kt6(fld, tower)),
        ("KT3", lambda: search_kt3(tower)),
        ("CUBE3Q", lambda: search_cube3q(tower)),
        ("PAIR3", lambda: search_pair3(fld, tower)),
    ):
        t0 = time.time()
        tables[name] = search()
        tm[name + "_s"] = time.time() - t0
    tables["LIN"] = tower.lin_tables()
    return tables, tm


# #########################################################################
# Models
# #########################################################################
# Bit-exact classical models of the circuit (Builder-free): the one Python
# model of its fused-Q BM recursion, and the Psi route (stages 5-7) op for
# op.  The Psi window search runs them.

# =========================================================================
# BM recursion (q-form conditional-swap, fused-Q B bank)
# =========================================================================
# The B bank rotates at bank-1.  A per-round hook instruments the loop.

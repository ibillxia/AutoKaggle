"""Exact op-type accounting per stage of the reference invsqrt circuit."""
import sys
sys.path.insert(0, 'code')
import env, kickmix as km
sys.path.insert(0, 'data/verifiers/invsqrt')
import nr_reference_circuit as R

GROUPS = ['CCX','CCZ','CSWAP','CX','CZ','X','Z','Z_POW','HMR','R','SWAP','NEG','BIT_STORE0','BIT_STORE1','BIT_INVERT','INFO_TAG']


def build(stages, prec=R.Precision(), n_p=10):
    width = 2 * n_p + 2
    out_bits = prec.output_bits(n_p)
    b = km.CircuitBuilder()
    x = b.create_quantum_register(width, name='x')
    y1 = b.alloc_qubits(out_bits); flags = b.alloc_qubits(n_p); sel = b.alloc_qubits(3)
    coefs = [b.alloc_qubits(prec.coef_bits) for _ in range(4)]
    sq = b.alloc_qubits(prec.square_bits); z = b.alloc_qubits(prec.product_bits + 2)
    scr = b.alloc_qubits(max(prec.square_bits, prec.product_bits + 2, prec.result_bits + 1))
    regs = (x, flags, sel, coefs, sq, z, y1, scr)
    if 'normalize' in stages: R._normalize(b, False, x, flags)
    if 'select' in stages: R._select_interval(b, False, x, sel)
    if 'load' in stages: R._load_coefficients(b, prec, x, sel, coefs)
    if 'horner' in stages: R._horner(b, False, prec, x, sel, coefs, scr)
    if 'square' in stages: R._square(b, False, coefs[0], sq, scr)
    if 'mulm' in stages: R._multiply_m(b, False, x, sq, z, scr)
    if 'correct' in stages: R._correct(b, False, z)
    if 'muly' in stages: R._multiply_y0(b, False, coefs[0], z, y1[:len(y1)-len(flags)], scr)
    if 'denorm' in stages: R._denormalize(b, False, flags, y1)
    if 'phase' in stages: R._phase_by_bits(b, y1)
    return b.finish_circuit()


def counts(c):
    d = {k: v for k, v in c.operation_statistics().items()}
    return d

ALL = ['normalize','select','load','horner','square','mulm','correct','muly','denorm','phase']
prev = None
print(f"{'stage':10s} {'Q':>4s} {'dCCX':>6s} {'dCSWAP':>7s} {'dCX':>7s} {'dHMR':>7s} {'ops':>8s}")
for i, s in enumerate(ALL):
    c = build(ALL[:i+1]); d = counts(c)
    g = lambda k: d.get(k, 0) + d.get(k+'_IF', 0)
    if prev is None:
        print(f"{'<alloc>':10s} {c.num_qubits:4d} {g('CCX'):6d} {g('CSWAP'):7d} {g('CX'):7d} {g('HMR'):7d} {len(c):8d}")
        prev = {k: g(k) for k in GROUPS}; prevq = c.num_qubits
        continue
    cur = {k: g(k) for k in GROUPS}
    print(f'{s:10s} {c.num_qubits:4d} {cur["CCX"]-prev["CCX"]:6d} {cur["CSWAP"]-prev["CSWAP"]:7d} {cur["CX"]-prev["CX"]:7d} {cur["HMR"]-prev["HMR"]:7d} {len(c):8d}')
    prev = cur
print()
c = build(ALL)
d = counts(c)
print('full compute-direction totals:', {k: v for k, v in d.items() if v})

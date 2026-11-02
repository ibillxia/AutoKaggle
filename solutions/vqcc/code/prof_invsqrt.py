"""Per-stage Toffoli/CSWAP accounting for the reference invsqrt circuit."""
import sys, dataclasses
sys.path.insert(0, 'code')
import env, kickmix as km
sys.path.insert(0, 'data/verifiers/invsqrt')
import nr_reference_circuit as R


def stats(build):
    c = build()
    s = c.operation_statistics()
    d = {k: v for k, v in s.items()}
    return c.num_qubits, d.get('CCX', 0), d.get('CSWAP', 0) + d.get('CSWAP_IF', 0), d.get('CCZ',0), d.get('Z_POW',0)


def build_all(stages, prec=R.Precision(), n_p=10):
    width = 2 * n_p + 2
    out_bits = prec.output_bits(n_p)
    b = km.CircuitBuilder()
    x = b.create_quantum_register(width, name='x')
    y1 = b.alloc_qubits(out_bits)
    flags = b.alloc_qubits(n_p)
    sel = b.alloc_qubits(3)
    coefs = [b.alloc_qubits(prec.coef_bits) for _ in range(4)]
    sq = b.alloc_qubits(prec.square_bits)
    z = b.alloc_qubits(prec.product_bits + 2)
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


ALL = ['normalize','select','load','horner','square','mulm','correct','muly','denorm','phase']
prev = (22, 0, 0, 0, 0)
print(f"{'stage':12s} {'Q':>5s} {'dCCX':>7s} {'dCSWAP':>7s} {'cum_CCX':>8s}")
for i, s in enumerate(ALL):
    q, ccx, csw, ccz, zp = stats(lambda: build_all(ALL[:i+1]))
    print(f'{s:12s} {q:5d} {ccx-prev[1]:7d} {csw-prev[2]:7d} {ccx:8d}')
    prev = (q, ccx, csw, ccz, zp)

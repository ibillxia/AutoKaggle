"""Sweep Precision() widths against the exact fixed-point model (ground truth for the circuit)."""
import sys, itertools, math, time
sys.path.insert(0,'code')
import env
sys.path.insert(0,'data/verifiers/invsqrt')
import numpy as np
import nr_reference_circuit as R

N_P = 10
MAX_X = 3*(2**N_P-2)**2
XS = np.arange(1, MAX_X+1)
base = R.Precision()

def try_prec(d, wc, wsq, wz, wp):
    p = R.Precision(delta_bits=d, coef_bits=wc, square_bits=wsq, product_bits=wz, result_bits=wp)
    try:
        e = R.relative_error(N_P, p, XS)
    except OverflowError as ex:
        return None, str(ex)
    return float(e.max()), None

print(f'baseline: max rel err 2^{math.log2(R.relative_error(N_P, base, XS).max()):.2f}')
print()
print(f"{'d':>3s} {'wc':>3s} {'wsq':>4s} {'wz':>3s} {'wp':>3s} {'max_rel_err':>12s} {'log2':>7s} {'margin':>7s}")
best = []
for d in (8,10,12,14):
    for wc in (11,12,13,14,15,17):
        wsq = 2*wc; wz = 2*wc+2; wp = 28
        e, err = try_prec(d, wc, wsq, wz, wp)
        if e is None:
            print(f'{d:3d} {wc:3d} {wsq:4d} {wz:3d} {wp:3d}  OVERFLOW {err}')
            continue
        lg = math.log2(e) if e > 0 else -99
        ok = lg <= -24
        print(f'{d:3d} {wc:3d} {wsq:4d} {wz:3d} {wp:3d} {e:12.3e} {lg:7.2f} {"PASS" if ok else "fail":>7s}')
        if ok: best.append((lg, d, wc, wsq, wz, wp))

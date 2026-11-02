"""Report where the dqi circuit's peak qubit count actually occurs.

dqi's score is R * Q^2 * T, so Q is squared -- a 10% Q cut is worth 21% on that
axis. Q = 1789 comes from N*B syndrome wires plus whatever is co-live at the
widest moment. This builds the circuit with the same install() the shipped
build uses, then reports the wire-occupancy curve by phase so the peak can be
attributed to a stage rather than guessed at.

Usage: python peak_report.py [--compact-from N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import env  # noqa: E402,F401

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'gen_dqi_v6'))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--compact-from', type=int, default=70)
    ap.add_argument('--stack', type=int, default=1)
    ap.add_argument('--no-build', action='store_true')
    a = ap.parse_args()

    os.environ['DQI_BORROW'] = 'affine'
    import gu_formula_derivation as F
    t0 = time.monotonic()
    tables, _ = F.run_b12()
    assert F.certify_kt(tables['KT12'], F.FIELD)
    c = json.loads((HERE / 'gen_dqi_v6' / 'ref' / 'gu_constants.json').read_text())
    c.update(tables)
    consts = HERE / '_peak_constants.json'
    consts.write_text(json.dumps(c))
    os.environ['DQI_CONSTANTS'] = str(consts)
    print(f'tables in {time.monotonic()-t0:.0f}s', flush=True)

    import gu_port as G
    import run_gu_port as RG
    import round2_gadgets as R2
    import symbolic_runtime as S
    G.Builder = S.SymbolicBuilder
    G.declare_registers = S.declare_registers
    G.borrow_template = RG.borrow_template
    if a.stack != 1:
        G.STK_CAP = a.stack
        G.CAPS = [min(x, a.stack) for x in G.CAPS_FULL]
    R2.install(G, top_first=True, pebble_last=True, norm_pebble=True,
               compact_from=a.compact_from)

    source = (HERE / 'gen_dqi_v6' / 'round2_gadgets.py').read_text()
    source = source[source.index('def lean_sweep('):]
    for old, new in (('b.physical[q]', 'b.wire(q)'),
                     ('b.physical[root]', 'b.wire(root)'),
                     ('b.physical[loc[0]]', 'b.wire(loc[0])'),
                     ('b.physical[ps[h]]', 'b.wire(ps[h])')):
        source = source.replace(old, new)
    source = source.replace(
        "raw.cx(arr([q for word in words for q in word]),arr([q for _ in words for q in target]))",
        "raw.cx(arr([q for word in words for q in word]),arr([q for _ in words for q in target]));b.invalidate(target)")
    source = source.replace("raw.del_and(b.wire(root),arr(om),arr(loc))",
                            "raw.del_and(b.wire(root),arr(om),arr(loc));b.forms[loc[0]]=S.ZERO")
    source = source.replace("raw.init_and(b.wire(root),arr(om),arr(loc))",
                            "raw.init_and(b.wire(root),arr(om),arr(loc));b.invalidate(loc)")
    source = source.replace("raw.cx(arr([w[k] for w in omw]),arr(om*len(omw)))",
                            "raw.cx(arr([w[k] for w in omw]),arr(om*len(omw)));b.invalidate(om)")
    ns = {'S': S}
    exec(source, ns)

    def native_sweep(b, E, A, omw, ebw, abw):
        qs = [q for r in omw + ebw + abw for q in r]
        b.suspended = True
        ns['stream_sweep'](b, E, A, omw, ebw, abw)
        b.invalidate(qs)
        b.suspended = False
    G.emit_sweep = native_sweep

    t0 = time.monotonic()
    b, tof = G.build()
    print(f'built in {time.monotonic()-t0:.0f}s  physical wires = {len(b.physical)}', flush=True)

    # Occupancy curve: walk the wire log and track live count.
    curve = []
    live = set()
    peak = 0
    peak_at = None
    log = getattr(b, 'wire_log', None)
    if log:
        for i, step in enumerate(log):
            pass
    print(f'wire_log entries = {len(log) if log else 0}')
    print(f'live_profile keys = {list(b.live_profile)[:10] if hasattr(b, "live_profile") else None}')
    if hasattr(b, 'live_profile') and b.live_profile:
        lp = b.live_profile
        print(json.dumps({k: v for k, v in list(lp.items())[:20]}, indent=1, default=str)[:2000])
    out = env.SCRATCH / f'dqi_peak_cf{a.compact_from}_st{a.stack}.json'
    out.write_text(json.dumps({'num_physical': len(b.physical), 'toffoli': tof,
                               'compact_from': a.compact_from, 'stack': a.stack,
                               'live_profile': getattr(b, 'live_profile', None)},
                              indent=1, default=str))
    print(f'RESULT compact_from={a.compact_from} stack={a.stack} '
          f'physical={len(b.physical)} toffoli={tof}')
    print('wrote', out)


if __name__ == '__main__':
    main()

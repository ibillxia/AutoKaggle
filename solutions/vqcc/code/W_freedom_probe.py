"""Measure what the submitted H2_I (i.e. W_rc) does to Lambda, E_gap, lambda_eff.

The verifier freezes beta_1 and S but NOT H2_I: W_rc = H2_I - sum_b w_b is a
submitted array. It enters three places:

  lambda_h2 = 0.25 sum_rc (|H2_I| + L1(w_rc))^2
  h1_prime  = h1_thc - (eta/2) S + beta_1 I - sum_rc W_rc X_rc
  E_SOS    -= 0.5 ||W||^2

so Lambda and E_gap can move without touching the circuits at all. This sweeps
a family of W candidates and reports the verifier's own metrics for each.

Usage: python W_freedom_probe.py [SUBMISSION_NPZ]
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

FLOOR_LAM, FLOOR_GAP = 49.0, 3.914898


def metrics_with(params: dict, words, edges, brot: int) -> dict:
    """Run the verifier's own word evaluation on a modified parameter set."""
    inp = V.load_inputs(params)
    return V.evaluate_words(inp, words, edges, brot)


def main() -> None:
    npz = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz')
    z = np.load(npz)
    base = {k: z[k] for k in z.files}
    words, edges = z['angle_words'], z['tree_edges']
    w = np.asarray(z['W'], dtype=np.float64)
    H2 = np.asarray(z['H2_I'], dtype=np.float64)
    sum_b = w.sum(axis=1)

    variants = {
        'as submitted': H2,
        'H2_I = 0': np.zeros_like(H2),
        'H2_I = -sum_b w': -sum_b,
        'H2_I = 0.5*H2': 0.5 * H2,
        'H2_I = -0.5*sum_b w': -0.5 * sum_b,
    }
    rows = []
    for name, h2 in variants.items():
        p = dict(base)
        p['H2_I'] = h2
        try:
            m = metrics_with(p, words, edges, 10)
        except Exception as e:                       # noqa: BLE001
            rows.append({'variant': name, 'error': f'{type(e).__name__}: {e}'})
            continue
        W = h2 - sum_b
        rows.append({
            'variant': name,
            'error_1_mHa': 1e3 * m['error_1'],
            'error_2_mHa': 1e3 * m['error_2'],
            'Lambda': m['Lambda'],
            'E_gap': m['E_gap'],
            'lambda_eff': m['lambda_eff'],
            'gap_in_domain': bool(m['gap_in_domain']),
            'half_norm_W_sq': 0.5 * float(np.sum(W ** 2)),
            'passes': bool(V.passes_accuracy(m) and m['gap_in_domain']),
        })
    hdr = f"{'variant':>22}{'err1':>9}{'err2':>9}{'Lambda':>9}{'E_gap':>8}{'lam_eff':>9}{'pass':>6}"
    print(hdr)
    for r in rows:
        if 'error' in r:
            print(f"{r['variant']:>22}  {r['error']}")
            continue
        print(f"{r['variant']:>22}{r['error_1_mHa']:>9.2f}{r['error_2_mHa']:>9.2f}"
              f"{r['Lambda']:>9.3f}{r['E_gap']:>8.4f}{r['lambda_eff']:>9.3f}"
              f"{str(r['passes']):>6}")
    (env.SCRATCH / 'w_freedom.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()

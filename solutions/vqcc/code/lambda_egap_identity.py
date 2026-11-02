"""Is Lambda pinned by the E_gap gate? Measure, do not assume.

E_gap = E_upper - e_sos, and e_sos = A - (Lambda_h1 - tr(h1_prime)) - 0.5*||W||^2.
With tr(h1_prime) - Lambda_h1 = -2p (p = |sum of negative eigenvalues|),

    E_gap = (E_upper - A) + 2p + 0.5*||W||_F^2 .

Meanwhile Lambda = ||h1_prime||_nuc + 0.25*sum_rc (L1_rc + |W_rc + sum_b w_b|)^2.

If |W_rc + sum_b w_b| ~ |W_rc| then the h2 part is ~0.25*||W||_F^2, i.e. HALF the
quantity E_gap forces to be 563.047371. This script measures that overlap and the
resulting slack, so the claim 'Lambda is pinned' is a number, not a hunch.

Usage: python lambda_egap_identity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as M  # noqa: E402


def main() -> None:
    npz = env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz'
    z = np.load(npz)
    inp = V.load_inputs(npz)
    met = V.evaluate_words(inp, z['angle_words'], z['tree_edges'], brot=V.KEEP_BITS)
    W = np.asarray(inp.W, dtype=np.float64)      # (R, C)
    w = np.asarray(inp.w, dtype=np.float64)      # (R, B, C)
    wB = W + w.sum(axis=1)
    l1 = np.abs(w).sum(axis=1)

    lam_h2 = float(0.25 * np.sum((l1 + np.abs(wB)) ** 2))
    print(f"Lambda            {met['Lambda']:.6f}")
    print(f"  lambda_h2       {lam_h2:.6f}  ({100 * lam_h2 / met['Lambda']:.2f}%)")
    print(f"  lambda_h1       {met['Lambda'] - lam_h2:.6f}")
    print(f"E_gap             {met['E_gap']:.6f}   (floor {V.E_GAP_FLOOR})")
    print(f"lambda_eff        {met['lambda_eff']:.6f}")
    print()
    print(f"0.5*||W||_F^2     {0.5 * (W ** 2).sum():.6f}")
    print(f"0.25*||W||_F^2    {0.25 * (W ** 2).sum():.6f}   <- would BE lambda_h2 "
          f"if H2_I ~ W")
    print(f"0.25*||H2_I||_F^2 {0.25 * (wB ** 2).sum():.6f}")
    print(f"0.25*||L1||_F^2   {0.25 * (l1 ** 2).sum():.6f}")
    print(f"cross 0.5*sum l1*|H2_I|  {0.5 * float(np.sum(l1 * np.abs(wB))):.6f}")
    print()
    # How much does H2_I deviate from W?  That deviation is the only way lambda_h2
    # can fall below 0.25*||W||^2, and it costs E_gap one-for-one.
    dev = float(np.sum((wB - W) ** 2))
    print(f"||H2_I - W||_F^2  {dev:.6f}   (= ||sum_b w_b||_F^2)")
    print(f"ratio ||H2_I||/||W||  {np.linalg.norm(wB) / np.linalg.norm(W):.4f}")
    print()
    # The trade: lowering lambda_h2 below 0.25||W||^2 needs H2_I -> 0, but E_gap
    # needs 0.5||W||^2 large.  Enumerate the best possible Lambda at each E_gap.
    print('Best achievable pair if H2_I were driven to exactly 0:')
    print(f"  then lambda_h2 = 0.25*||L1||^2 = {0.25 * (l1 ** 2).sum():.4f},")
    print(f"  but E_gap loses 0.5*||W||^2 = {0.5 * (W ** 2).sum():.4f} unless W is")
    print(f"  re-supplied, which puts it straight back into lambda_h2.")
    print()
    print(f"Slack if Lambda could sit at its E_gap-forced value:")
    print(f"  lambda_eff at E_gap = floor with Lambda unchanged: "
          f"{np.sqrt(V.E_GAP_FLOOR * (2 * met['Lambda'] - V.E_GAP_FLOOR)):.5f} "
          f"(x{met['lambda_eff'] / np.sqrt(V.E_GAP_FLOOR * (2 * met['Lambda'] - V.E_GAP_FLOOR)):.6f})")


if __name__ == '__main__':
    main()

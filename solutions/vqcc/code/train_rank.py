"""Train the FeMoCo DF-THC factorization at a chosen (rank, bases, copies), seeded from the baseline.

Lower rank means fewer angle rows, so the ANGLE circuits -- most of the chemistry Toffoli count --
shrink. Whether the 20 mHa accuracy gates still hold is the question this answers.

Usage: python train_rank.py RANK BASES COPIES EPOCHS [--lr X] [--out PATH]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
KIT = ROOT / 'data' / 'starter' / 'kit' / 'starter_kit'
sys.path.insert(0, str(KIT / 'starter' / 'task1_factorization'))
sys.path.insert(0, str(KIT))
import env  # noqa: E402,F401

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import train_df_thc_qaggle as T  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('rank', type=int)
ap.add_argument('bases', type=int)
ap.add_argument('copies', type=int)
ap.add_argument('epochs', type=int)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--rho', type=float, default=1e-4)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--print-interval', type=int, default=200)
args = ap.parse_args()

fcidump = str(KIT / 'data' / 'integrals' / 'FeMoCo76o113e.fcidump')
seed = str(KIT / 'baselines' / 'FeMoCo76o113e_65_60_60_ep50000_seed0.npz')
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    raise SystemExit(f'{out} exists; refusing to overwrite')

sol = T.train_df_thc(
    fcidump_file=fcidump,
    upper_energy=T.SYSTEMS['FeMoCo76o113e']['upper_energy'],
    rank=args.rank, bases=args.bases, copies=args.copies,
    epochs=args.epochs, lr=args.lr, rho=args.rho, seed=args.seed,
    print_interval=args.print_interval,
    seed_solution_file=seed,
    output_file=None,
)
# train_df_thc returns (params, history); save the verifier-format npz so the run can be
# scored with chem_score.py afterwards.
params = {k: np.asarray(v) for k, v in sol[0].items()}
np.savez(out, **params)
print('wrote', out, {k: v.shape for k, v in params.items()})

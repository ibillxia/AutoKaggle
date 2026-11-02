"""Train FeMoCo76o113e at a small (bases, copies) and score every checkpoint locally.

BACKGROUND RUN. The point is to find the smallest B whose trained factorization still
holds error_1, error_2 <= 20 mHa and the E_gap floor. Chemistry is the only axis that can
move the composite score, and T_angles = 2*K*(N + R*B) is linear in B.

Each epoch prints a line; each checkpoint is re-scored with the OFFICIAL numpy evaluator
(dfthcsos.scoreinfra.dfthc_math), not the trainer's internal numbers, so a PASS here means
the factorization is genuinely admissible.

Usage: python train_small.py --bases 30 --copies 30 --epochs 20000 --out RUN_DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import env  # noqa: E402,F401

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

KIT = env.DATA / 'starter' / 'kit' / 'starter_kit'
sys.path.insert(0, str(KIT / 'starter' / 'task1_factorization'))
sys.path.insert(0, str(KIT))

import numpy as np  # noqa: E402
import train_df_thc_qaggle as T  # noqa: E402
from dfthcsos.verify import heuristic_costs, load_inputs  # noqa: E402
from dfthcsos.scoreinfra.dfthc_math import (  # noqa: E402
    evaluate_dfthc_vectors,
    reconstruct_u_vectors,
    split_u_vectors,
)

N = 76


J = 10


def encode_words(vectors, edges, brot):
    """Givens-tree angle words for each row, quantized to brot bits (as the verifier reads)."""
    v = np.asarray(vectors, float).copy()
    a = np.zeros((len(v), len(edges)))
    for j in reversed(range(len(edges))):
        p, c = edges[j]
        a[:, j] = np.arctan2(v[:, c], v[:, p])
        v[:, p] = np.hypot(v[:, p], v[:, c])
        v[:, c] = 0.0
    return np.rint(a * (2 ** (brot - 1) / np.pi)).astype(np.int64) % (1 << brot)


def official_score(params, tree, brot):
    """Score a factorization with the verifier's own math; returns metrics + cost."""
    inp = load_inputs(params)
    U = np.asarray(params['U'], float)
    R, B, _ = U.shape
    u_all = U.reshape(R * B, N)
    # The N one-body rows are independent of B; take them from the verified source.
    inp0 = load_inputs(SRC / 'submission.npz')
    v0 = reconstruct_u_vectors(WORDS0 / (1 << (brot - 1)), TREE0)
    v0 = v0 / np.linalg.norm(v0, axis=1, keepdims=True)
    one_body, _ = split_u_vectors(v0, inp0.w)
    u_all = np.concatenate([one_body, u_all], axis=0)
    words = encode_words(u_all, tree, brot)
    # Score what the circuits would actually load: the quantized words, decoded back.
    m = evaluate_dfthc_vectors(
        reconstruct_u_vectors(words / (1 << (brot - 1)), tree),
        inp.w, inp.W, inp.S, inp.beta_1, inp.ref, energy_upper=inp.energy_upper)
    R, B, C = inp.w.shape
    K = -(-(N - 1) // J)
    angles_toffoli = 2 * K * (N + R * B)
    h = heuristic_costs(N, R, B, C, brot)
    q = h['persistent_qubits'] + J * brot + max(h['temporary_qubits'], 11)
    t = angles_toffoli + h['toffoli']
    return m, h, q, t, m['lambda_eff'] * t * q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bases', type=int, required=True)
    ap.add_argument('--copies', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=20000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--rho', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed-file', type=Path, default=None)
    ap.add_argument('--save-every', type=int, default=2000)
    ap.add_argument('--print-interval', type=int, default=250)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    # Angle words and tree come from the current verified submission: the ANGLE circuits
    # load N-1 tree rotations per vector and are rebuilt later once a factorization passes.
    sub = ROOT / '_scratch' / 'submission' / 'dfthcsos'
    P = dict(np.load(sub / 'submission.npz', allow_pickle=False))
    words, tree, brot = P['angle_words'], P['tree_edges'], 10
    global SRC, WORDS0, TREE0
    SRC, WORDS0, TREE0 = sub, words, tree
    ref_b, ref_c = P['W'].shape[1], P['W'].shape[2]
    # T_angles = 2 * K * num_vectors with K = ceil(75/J); recomputed per checkpoint.

    seed_file = args.seed_file
    if seed_file is None:
        seed_file = (KIT / 'baselines' / 'FeMoCo76o113e_65_60_60_ep50000_seed0.npz')
    log = args.out / 'log.jsonl'
    print(f'run bases={args.bases} copies={args.copies} epochs={args.epochs} '
          f'seed={seed_file.name}', flush=True)

    prev = seed_file
    t0 = time.time()
    for chunk in range(0, args.epochs, args.save_every):
        sol = T.train_df_thc(
            fcidump_file=str(KIT / 'data' / 'integrals' / 'FeMoCo76o113e.fcidump'),
            upper_energy=T.SYSTEMS['FeMoCo76o113e']['upper_energy'],
            rank=65, bases=args.bases, copies=args.copies,
            epochs=min(args.save_every, args.epochs - chunk),
            lr=args.lr, rho=args.rho, seed=args.seed,
            print_interval=args.print_interval,
            seed_solution_file=str(prev), output_file=None,
        )
        params = {k: np.asarray(v) for k, v in sol[0].items()}
        ckpt = args.out / f'ckpt_{chunk + args.save_every}.npz'
        np.savez(ckpt, **params)
        try:
            m, h, q, t, cost = official_score(params, tree, brot)
            rec = {'epoch': chunk + args.save_every, 'seconds': round(time.time() - t0, 1),
                   'error_1': m['error_1'], 'error_2': m['error_2'],
                   'Lambda': m['Lambda'], 'E_gap': m['E_gap'],
                   'lambda_eff': m['lambda_eff'], 'gap_in_domain': m['gap_in_domain'],
                   'Q': q, 'T': t, 'cost': cost}
            rec['pass'] = bool(m['error_1'] <= 0.020 and m['error_2'] <= 0.020
                               and m['gap_in_domain'] and m['Lambda'] > 49.0)
        except Exception as exc:                      # noqa: BLE001
            rec = {'epoch': chunk + args.save_every, 'error': repr(exc)}
        with log.open('a') as fh:
            fh.write(json.dumps(rec) + '\n')
        print('CHECKPOINT ' + json.dumps(rec), flush=True)
        prev = ckpt                      # continue from here, do not restart
        if rec.get('pass'):
            print(f'*** FEASIBLE factorization at B={args.bases}: {ckpt}', flush=True)
            break


if __name__ == '__main__':
    main()

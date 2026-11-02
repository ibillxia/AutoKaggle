# vqcc (Kaggle "Google Quantum AI - Verifiable Quantum Circuit Compilation") — facts established locally

Everything below was verified on this machine unless marked otherwise.

## Competition

- `kaggle.com/competitions/google-verifiable-quantum-circuit-compilation`, id **161954**,
  Featured, organizer Google Quantum AI, prize pool **$150,000** (6 prizes).
- Enabled 2026-10-22, deadline **2027-01-20 23:59 UTC**; team merge / new entrant deadline
  2027-01-13. Max team size 5. **5 submissions/day**, 2 scored submissions kept.
- Submission file must be **`submission.zip`** (≤ 20 GB), containing three entries:
  `invsqrt.km[xb]`, `dqi.km[xb]`, `dfthcsos/` (angle circuits + `submission.npz`).
- Metric "Google Quantum AI Metric", **50% public / 50% private** split. Notebooks allowed
  (CPU 540 min, GPU 540 min), internet allowed, identity verification required.
- Rules allow public external data/tools; **winners must open-source under Apache-2.0**.
- The spec paper is `verifiable_circuit_compilation.pdf` = **arXiv:2610.23832v2**
  ("Leveraging verifiable structure in quantum circuits to find more efficient
  compilations with agentic AI", Khattar, Rubin, Gidney, Broughton, Low, Babbush).

## How scoring works (read this before optimizing anything)

Each benchmark's verifier returns a **cost** (lower is better). The leaderboard score is

```
score = geometric mean over the 3 benchmarks of (baseline_cost / your_cost)
```

The hosts' reference submission (`sample_submission/`) scores exactly **1.0**;
halving every cost scores 2.0. All three are kickmix circuits verified by simulation,
so **the game is pure circuit synthesis** — there is no training data, no test set.

Verified locally with the official verifiers (see `artifacts/baseline_costs.json`):

| task | baseline cost | composition |
|---|---|---|
| invsqrt | **2,076,774.23** | Q=260 × C=7987.59 (7735 Toffoli + 1 T + 37 rotations) |
| dqi | **24,439,463,583,672.93** | R(ε₊)=1.19126 × Q²=1893² × T=5,725,108.38 |
| dfthcsos | **3,355,599,588.32** | λ_eff=49.30137 × T=93,237 × Q=730 |

Round-trip check: the paper's numbers (7,735 Toffoli/260 qubits; 5,725,461 Toffoli/1,893
qubits; 93,237/730) all reproduce exactly, so the local build is faithful.

## Leaderboard bar (public, snapshot 2026-10-23)

| rank | score |
|---|---|
| 1 Scott Willis | 10.96163 |
| 5 pjmathematician | 9.92735 |
| 10 Pavel | 8.23300 |
| **25 JezzLynn** | **6.24970** ← ≈ top 10% line (256 teams) |
| 34 Marko Rupnik | 5.82282 |
| 45 1m0ke | 5.51687 |

256 teams at snapshot. **Aim for ≳7**, not 6.25, because the bar drifts.

Best *public* notebook at the time of writing projects **5.64**
(ratios 11.85 invsqrt × 10.58 dqi × 1.43 chemistry). So the public state of the art is
below the 10% line, and the chemistry benchmark is where the public is weakest (1.43×).

## The three tasks

All circuits are **kickmix** circuits: only reversible classical gates (X, CX, CCX, SWAP,
CSWAP), diagonal phases (Z, CZ, CCZ, Z_POW, NEG), measurement/reset (HMR, R) and classical
bit ops. Each maps a basis state to a basis state times a phase, so one CPU core can
simulate 10⁸ gates in seconds. Verification is fuzz testing with Fiat–Shamir seeding
(seed = SHA-256 of the submitted circuit ⊕ a secret salt).

### I. invsqrt — phase by 1/√x

`|x>|0> → e^{2πi/√x}|x>|0>` for every x ∈ [1, 3,133,452], x is a 22-qubit register named
`x` (little-endian), everything else is scratch that must return to |0>. 64 shots per x
(200,540,928 shots), tolerance `2^-24/√x` turns. Cost **Q·C**, C = Toffoli + T/4 with
rotations priced `0.56·log₂(1/ε)+5.3` T.

### II. DQI — Reed–Solomon syndrome decoder

Registers `S0..S69` (70 × 12 qubits, the syndrome) and `E0..E4094` (4095 × 12 classical
bits, the MBUC record c); no other named registers. Restore `S`, clean all scratch, apply
phase `(-1)^{Σ popcount(e_j ∧ M_j)}` where e is the unique weight-≤35 error with syndrome
s. 102,400 shots; failures allowed but priced. Cost **R(ε₊)·Q²·T**, R(0)=1.191 floor,
ε₊ ≥ 1/16 rejected.

### III. dfthcsos — FeMoCo DFTHC-SOS block encoding

N=76 orbitals, η=113 electrons. Submission = factorization arrays + Givens tree +
K = ceil(75/J) angle-loading circuits, each ≤ **128 qubits**, `selection` = 12 qubits,
`word_0..word_{J-1}` = brot bits each. Floors: raw Λ > 49.0, E_gap > 3.914898 Ha,
error₁,error₂ ≤ 0.020 Ha. Cost **λ_eff·T_total·Q_total**.

## Local validation harness (this is the enabler)

`data/kickmix/kickmix.so` is a **Linux ELF** — unusable on Windows. It is rebuilt from the
official Qualtran source at commit `c58703d0` (the same commit the dataset was built from)
with MSYS2 UCRT g++ via `code/build_kickmix.sh`. Build-only adaptations:

- `std::aligned_alloc` (absent from MinGW headers) → `_aligned_malloc`, and the matching
  releases → `_aligned_free`, in the three (four) affected files;
- `cmd/` tools and `arg_parse.cc` excluded (CLI only, and `arg_parse.h` lacks `<cstdint>`).

Two environment traps, both solved in `code/env.py`:

1. `Circuit.to_file`/`from_file` round-trip through C `tmpfile()`, which fails with EACCES
   here because the user's `%TEMP%` is not writable for these processes. `env.py` points
   TMP/TEMP at `solutions/vqcc/_scratch` **before** importing kickmix.
2. `libwinpthread-1.dll` must sit next to `kickmix.pyd` (copied at build time).

Python is **3.14.x** (uv venv): 3.12 is refused by kickmix's `PyLong_AsNativeBytes` usage,
and uv-built 3.13 environments could not write `%TEMP%` at all.

Timings on this machine (i7-14700F, 28 threads, 32 GB):

| run | wall clock |
|---|---|
| invsqrt, 1 shot per x (3.13 M shots) | 5 m 45 s |
| invsqrt, full 64 shots per x | ~6 h (extrapolated) |
| dqi, full 102,400 shots | 3 m 30 s |
| dfthcsos, full 64 shots/address | 36 s (+ ~10 s FCIDUMP parse) |

Simulator throughput measured at ~7.4 × 10⁵ shots/s for the 62 k-op invsqrt circuit, so
`invsqrt`'s cost is dominated by the *Python* per-batch bookkeeping (48961 chunks), not the
simulation. A fast local checker that samples a subset of x is the practical iteration
loop; the full run is only for pre-submission confidence.

## Repository layout

```
solutions/vqcc/
  code/            env.py, score.py, run_invsqrt.py, run_dqi.py, baselines.py,
                   build_kickmix.sh, compat/
  data/            official competition data (sample_submission, verifiers, kickmix,
                   starter kit) — not in git
  build/           kickmix.pyd + patched source copy — not in git
  .venv/           python 3.14 + numpy + scipy + pybind11 — not in git
  artifacts/       baseline_costs.json and future measurements
  _scratch/        temp dir for kickmix's tmpfile(); unpacked submissions
```

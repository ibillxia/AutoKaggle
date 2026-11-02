# vqcc — optimization scorecard (2026-10-24)

Everything below is measured locally with the official verifiers, not projected.
Where a claim was refuted, the refutation is recorded so it is not retried.

## Where we stand

Current best, all three components verified locally on 2026-10-28 with the official
verifiers from the exact bytes packaged into `_scratch/submission.zip`
(18 members, SHA-256 `4245a5ca…`, 25,886,827 B), submitted as ref 57739663:

| task | our cost | ratio | status |
|---|---:|---:|---|
| invsqrt | 102,664.240 | **20.2288×** | busyaprime's re-synthesised lookup in mrmeows' 63-qubit build, PASS (TE 1,629.59, toffoli 1,334.50, 41 rotations) |
| dqi | 4.147813e11 | **58.9213×** | heliosli's 889-qubit decoder, PASS (T 440,563.96, R 1.19126, 102,400 shots) |
| dfthcsos | 3.431726e8 | **9.7782×** | somejoint R8/B143/C60 XOR2, PASS (T 16,366, Q 430, λ_eff 48.7643, error₂ 19.9988 mHa) |

**Composite: 22.6725 locally; the leaderboard scored the same archive 22.67253**
(ref 57739663, COMPLETE) — the public board on 2026-10-28 14:36 has 1,419 rows, which puts
22.67253 at about rank 68, top 4.8%, against a top-10% cut of 22.6297. Twelve or so teams
hold 22.68–22.71, i.e. small private gains on this same public frontier; the leader is at
26.1411.

| archive | local composite | leaderboard |
|---|---:|---:|
| v9 (ref 57734440) | 21.8462 | 21.87960 (rank 536/1398, top 38.3%) |
| 22.63 portfolio (ref 57738300) | 22.6297 | 22.62971 (rank 94/1414, top 6.6%) |
| **22.67 portfolio (ref 57739663)** | **22.6725** | **22.67253 (rank ~68/1419, top 4.8%)** |
| v2 (ref 57598315) | 6.7267 | 6.68900 |

### 2026-10-29 late: the shipped portfolio, and the public frontier closed on all three tasks

Best archive, all three components re-verified locally from the packaged bytes and then
confirmed by the board (ref 57755033, COMPLETE, **22.78244**; local prediction 22.782443,
agreement to 5 decimals):

| task | shipped file | our cost | ratio | note |
|---|---|---:|---:|---|
| invsqrt | `9a8627c8` | 102,349.236 | 20.2907× | TE 1,624.59, toffoli 1,329.50, 41 rotations, 63 q |
| dqi | `53d8b2ed` | 4.12943e11 | 59.1829× | 897 q, 0/0 errors |
| dfthcsos | mixed bundle | 340,781,040.24 | 9.8470× | λ 48.632428, T 16,296, Q 430 |

**Composite 22.78244 → rank 77.** The two ladders below it scored exactly as predicted
(57754706 → 22.75911, 57754666 → 22.72544), so the local harness is calibrated to <1e-5.

Medal lines moved sharply during the day: silver (rank 71) went 22.74106 → **22.80026**,
bronze (rank 143) is 22.72080. We are 6 places and **0.0178 composite** short of silver.

`harvest.py` confirms we hold the public frontier on every task: our invsqrt `9a8627c8` beats
every published invsqrt (next best 102,664.24), our dqi `53d8b2ed` beats every published dqi
(next 414,004,704,471), and the chemistry mixed bundle beats every one of the 52 published
bundles (best single 340,990,159.68).

#### The chemistry cost is `λ · T · Q`, and only `T` is movable

Read off `dfthcsos/verify.py`:

```python
Q_total           = outer_persistent + inner_persistent     # SpinFreeData_V1(...)
heuristic_toffoli = outer 369 + inner 5311 + givens 2250 = 7930
T_total           = angles_toffoli + heuristic_toffoli
```

Both `Q` and the 7,930 are functions of the factorization shape only, not of the circuits.
For R8/B143/C60/brot=8 that pins `Q = 126 persistent + 304 temporary = 430`. (Q is not
purely shape-fixed across the field — public bundles span 415…681 — but every `Q < 430`
option measured pays ≥ +12% T, e.g. the best 415-q bundle costs 436,969,552.)

**λ is closed.** Eight iterations of exact-metric LP over all 488 variables (w 143×60×8 and
w_B), with analytic gradients verified against central differences, moved λ_eff from
48.6324280236 to 48.6324267882 — **−2.5e-8 relative**. All three gates sit at equality:

| gate | value | limit | slack |
|---|---:|---:|---:|
| error_1 | 0.0199976000 | 0.020 | 2.4e-6 |
| error_2 | 0.0199976000 | 0.020 | 2.4e-6 |
| E_gap | 3.9148994001 | 3.914898 | 1.4e-6 |

The LP's first-order optimum is large (dλ_eff = −17.9 over a ±100% box), so the landscape is
not flat — but every first-order descent direction *raises* error_1, whose slack is exhausted
at step t ≈ 6e-8. A nonlocal 2-parameter scan (w = a·w₀, w_B = b·w₀B, 441 points) found
exactly one feasible point: the start itself. Fixed-word family exhausted.

**T is at the measured floor of the entire public field.** The verifier charges ANGLE =
2 × static CCX count. Our per-batch vector is

```
[302,302,302,302,280,259,280,259,280,259,280,259,280,259,280] = 4,183 CCX
```

Across all 52 bundles (full 15-circuit sets for the 12 with T_angles ≤ 8,436; k=4/k=5 for the
other 40), the elementwise minimum **is our vector**. Both 8,366-total bundles ship
byte-identical odd-k files, and exactly one compilation in the whole field reaches 259.
Peephole is empty on all 15 circuits: 0 adjacent duplicate Toffoli pairs, 0 identical Toffoli
within 4 ops, 0 repeated-qubit Toffolis. The 15-way split is not free either —
`recover_words_from_circuits` forces `len(circuits) = ceil(75/J)` and circuit k must implement
XOR(batch k−1, batch k); J=6 needs 48 word qubits but our best compilation already uses 126 of
the 128 cap, and J=4 costs 9,840–10,362 T_angles.

**Remaining gap to silver: −38.5 charged Toffolis = −19.25 CCX out of 4,183 (−0.46%).**
That needs the word table re-synthesised, not selected — the same class of work as the
invsqrt constant-seed compiler.

#### invsqrt: closed within the architecture

Three doors, each with numbers. Qubits: peak liveness is a flat 63 across ops 4145–36088
(22 for x, 41 scratch simultaneously live); 1 qubit is worth 25.8 Toffolis and mrmeows' own
64-qubit variant saved only 7. Rotations: the lowest tap 2⁻³⁵ turn against a 2⁻³⁴·⁸ turn
tolerance is exactly saturated (worst error/tol 0.99932 at x=161), and 41→33 "cost more
arithmetic than it saved". Toffolis: 194 CSWAPs are already minimal Fredkins, 327 repeated
triples are compute/uncompute pairs, adjacent self-inverse pairs are all Clifford.

The one real opening is the author's **unbuilt constant-seed model** (128-term seed table,
8-bit decimals, degree 3, residual 24+2 guard, 26-bit output, error/tol 0.617 over the whole
domain), which removes the normalization + Newton–Raphson block. Exchange rate: 1
Toffoli-equivalent = 0.0616% of the invsqrt cost. Compiling it under ~1,100 Toffolis would be
worth roughly +3–5% composite, i.e. 23.5–24.0 — silver with margin, possibly gold.

#### DQI: the two public improvements are real, orthogonal in cost, and unmergeable in bytes

DQI cost is `R(eps) · Q² · T`, so heliosli's 897→889 narrowing (−1.8%, "almost all from Q²")
and Agniv's convert/unconvert rebuild (−0.44%) compose in arithmetic. Nobody in the public
field ships both:

| bundle | Q | mean Toffolis | cost |
|---|---:|---:|---:|
| `53d8b2ed` Agniv v19 (**shipped**) | 897 | 430,822.9 | 4.129431e11 |
| `e697b818` Atakan v10 | 889 | 439,739.0 | 4.140045e11 |
| `4ee7a17b` heliosli v11 | 889 | 440,564.0 | 4.147813e11 |

889 qubits at 430,822.9 Toffolis would cost 4.056102e11 → composite **22.91892** (rank ~52).
Silver needs only ≤ 4.119759e11, i.e. T ≤ 437,575 at Q = 889 — 2,164 Toffolis off Atakan.

**Not reachable by editing.** Both circuits already sit exactly at their true minimum width:
greedy interval colouring of the per-qubit use intervals gives peak overlap 897 for ours
(declared 897) and 889 for heliosli's (declared 889), so a free relabel gains nothing. The
peaks are at opposite ends of the circuit (op 6,969,223 of 59,852,041 vs 48,543,265 of
55,779,138) — the two lineages use different scratch schedules, which is why they cannot be
spliced. Base `b3354f60e459` and heliosli's have byte-identical `@enter/exit` tag sequences
(17,690 tags), so the narrowing is intra-block and the op delta is only 993 ops (CCX +30,
CZ +21, HMR +30, X +864, Z +9, 9 zero-test chains), matching heliosli's description. But at
top level the two authors overlap in scan / unsynthesis / unextract / convert / unconvert,
and changing scratch in one block renumbers every downstream qubit. Text 3-way merge is out
too: base and heliosli agree at only 87.7% of positions despite differing by 993 ops.

This is a recompile, not an edit. `solutions/vqcc/code/dqi_src/` rebuilds SomeJoint's decoder
from source but yields **1,789 qubits**, nowhere near the 897-qubit public lineage, so it
cannot reproduce either.

**But the generator of the shipped circuit is public.** SomeJoint's
`quantum-circuits-22-75-local-gm-897-qubit-dqi` carries `MEMBERS['dqi.kmb']['sha256'] =
53d8b2edf251e2a0…`, byte-identical to the file we ship, and its tree is 84 files / 346 KB,
hash-verified against its own `SOURCE_HASHES`, with no dependency beyond `circ`/`kickmix` and
the stdlib. Width is not a knob — `build.py` is 42 lines:

```python
S = [b.alloc_qubits(12) for _ in range(70)]     # 840, structural
E = [b.alloc_bits(12) for _ in range(4095)]     # classical only, never enters Q
module.decode(b, S, E)                          # Q = 840 + 57 scratch
```

The 57 is the allocator's high-water mark. But the scratch schedule is a **module choice**,
and the tree already ships alternatives that state the exact exchange rate:
`exp/inverse_width/kernel.py` — "Reverse220T adds five Toffolis while reducing reverse
private workspace 42→39". That is heliosli's trade in miniature. The `exp/` cluster
(`inverse_width`, `inverse_width_subspaces`, `noscratch`, `phase24_tradeoff`,
`mask_projection`, `safe_scalar`, `alternating_gauge`, `convert_short_hankel`, `fold_*`)
holds ~40 modules beyond what the entry point imports. `convert_safe` is generated, and
`head_pivot` / `aligned_route` / `full_head_product` are already in this tree
(`exp/r-main/stream/direct_converter.py`, `measured_heads.py`) — Agniv's rebuild is a
different assembly of the same parts, not a foreign lineage.

So the merge that failed at the binary level becomes ordinary in source. Next step would be
to port the tree, rebuild once, and check the SHA against `53d8b2ed…` — a pure reproduction
that either matches or proves the tree incomplete.

**It matches.** The tree is ported to `solutions/vqcc/code/dqi_897/` (80 files) and rebuilt
with our own source-built `kickmix.pyd` under the venv's Python:

```
cd solutions/vqcc/code/dqi_897
PYTHONPATH=solutions/vqcc/build ../../.venv/Scripts/python.exe -X utf8 build.py OUT.kmb
{"sha256":"53d8b2edf251e2a0c54c401a97d1396c7ed309e1cbfaab8d54a2c99a11ed6552",
 "Q":897,"T_static":444667,"num_ops":59852041,"num_bits":103536,"seconds":296.5}
```

Byte-identical to the shipped `dqi.kmb`. So the generator is complete, the toolchain is
sound, and Q is now a source-level variable rather than a byte layout.

The narrowing budget is generous because cost goes as `Q²`: at Q = 889 we may add ~8,038
static Toffolis and break even, and ~6,900 and still take silver. heliosli bought those
8 qubits at the binary level for +30 static. The peak sits at op 6,969,223 of 59,852,041 —
early — whereas heliosli's 889-qubit peak is late (op 48.5M of 55.8M), so the binding stage
here is a different one than they solved.

#### Where the 57 scratch actually are, and why they cannot be measured away

`set_max_qubits` bisection gives a single peak point, at
`external_normalize_paired → multiply_forward99`: allocated 873 + 24 requested = 897. The
composition is **840 structural + 24 `D` + 9 `pole` zero flags + 24 product registers**.
The 9 `pole` values (`zero_flag(b, D)`, i.e. `[D=0]`) are built in the forward pass, stored
in a token, and consumed by `clear_flag` in the reverse pass — so they accumulate one per
homogeneous node processed (live count 865, 866, … 873 across the 17 `gi`). heliosli's 889
is exactly those 8 accumulated flags removed, which matches their "9 zero-test chains added".

**Measuring them does not work.** The obvious trick — `hmr(pole, pm[0]); free(pole)` after
`cswap(pole, N, D)`, then re-prepare with `push_condition(pm[0]); x(pole)` in the reverse —
gives exactly the prize shape: **Q = 889, T_static = 444,667, unchanged, zero Toffoli cost**.
The verifier rejects it:

```
status FAIL   effective error 0.9997 >= 1/16
value_errors 102220/102400   phase_errors 46/102400
```

The flag is a computational-basis value, but after `cswap(pole, N, D)` it is entangled with
N and D, so measuring mid-circuit collapses the state. `clear_flag` gets away with measuring
because it sits at the end of the reverse pass, after the entanglement is undone. This is a
principle-level dead end, not a wiring bug.

Two other routes measured zero: the `decode_truncated` path gives Q 897 at T_static 446,510
(worse), and `multiply_forward_noscratch` still allocates the 24 product registers
("noscratch" refers to the kernel, not the target).

To eliminate `pole` without measuring you must either recompute each flag in the reverse pass
— blocked, because the flag is consumed before `D` is restored to its forward value — or
reorder the `gi` in `external_extraction` so the deepest node is processed while fewer flags
have accumulated — blocked, because that order is fixed by the `cx(b, d, a, g['c'])` carry
chains. Both are algorithm-level changes, not mechanical substitutions.

Silver now needs only **Q ≤ 895** (two qubits, with +914 mean Toffoli of headroom), not 889.

### The 2026-10-29 forum pass: where the medal lines are, and the chemistry swap

**Board, measured from page 1 (200 rows) on 2026-10-29:**

| position | score | note |
|---|---:|---|
| leader | 26.14113 | Breaking1800 |
| gold (rank 12) | 23.72216 | |
| **silver (rank 71)** | **22.74106** | top 5% |
| **bronze (rank 143)** | **22.71205** | top 10% |
| best public blend | 22.70030 | **rank 200, 13 teams tied** |
| ours before today | 22.67253 | off page 1 (> rank 200) |

So the public plateau no longer medals: 22.70030 is exactly the 200th row. Bronze now
needs 22.71205, silver 22.74106 — 0.175% and 0.302% of composite above 22.67253.

**The four forum posts that mattered** (third-party content, read as data, claims re-checked
against our own measurements where they touch our numbers):

- `753527` — marginal value: `invsqrt cost = Q × C`, so **one qubit = 1.61% of that cost =
  +0.54% of the score**; on DQI and chemistry one qubit is 0.23% (+0.08%). Break-even rates
  26 Toffoli-equivalents per invsqrt qubit, 990 Toffolis per DQI qubit.
- `753403` — inside the 63-qubit invsqrt: 41 scratch qubits, at most 35 hold a 1 at any
  instant, "all 41 are needed, their lifetimes are staggered. Saving a qubit means re-packing
  lifetimes, not lowering a peak." Worst input x=161 at 0.99932 of tolerance, mean 0.21.
- `753633` — a bound that holds for **any** factorization: E_gap > 3.914898 forces
  Λ ≥ D/2 = 281.52, so **λ_eff ≥ 46.79**, about 4% under the public 48.7. All the remaining
  slack is triangle-inequality slack in λ_h2. Independently confirms our own finding that
  re-optimising the offsets `w_B` moves λ_eff by <1e-6% (we measured −0.0009% on a strict
  floor). **Chemistry can only be won on T × Q.**
- `753588` — the RezOps briefing, plus the DQI salt result: the verifier seeds every shot
  from SHA-256(circuit hash + salt), and the board behaves as if the salt is fixed, so a
  DQI score is a fixed draw *per circuit file*, not per submission. Our `dqi.kmb`
  (`4ee7a17b4cbc`, 889 qubits) is that frontier file and drew 0 failures — the best possible
  R, expected final change −0.03%. Resubmitting identical bytes cannot re-roll it.

**The swap that closed most of the gap.** heliosli's dataset gained a version at
2026-10-29 00:53 whose chemistry bundle is stronger than the one the briefing describes
(it reports 16,306 Toffolis; this is 16,296). Downloaded into
`_tools/dl/heliosli-v12-20261029/`, verified with the official verifier straight out of the
staging directory and again out of the final zip:

```
status PASS   score 341133443.83596337   lambda_eff 48.68271909156811
toffoli 16296   qubits 430   num_ops 585673   reaction_depth 165
angles_toffoli 8366 (4183/pass)   angles_qubits 126
```

`npz_diff.py` against our shipped bundle: `U`, `angle_words` (1220, 75), `H1_I`,
`Hsym_lower`, `brot`, `tree_edges` are **identical**; only `W` and `H2_I` move. So it is the
same (8,143,60) factorization and the same words with a better weight refit
(λ_eff 48.7638 → 48.6827, −0.166%) *plus* a cheaper loader on 10 of the 15 batches
(T 16,366 → 16,296).

Note what that says about our own `wb_refit.py`: it only moved `H2_I` at fixed `W`. The
lever heliosli used is a refit of **all** of `W` at fixed words, which moves `X`, and with it
`h1_prime`, both errors and `E_gap`. That is inside the ≤0.3% that the forum's null list
allows for fixed-factorization refits, and it is the lever we did not pull.

Composite: **22.717621** (`score_chem.py`), +0.0450 on 22.672601 — estimated rank ≈ 95–100,
inside bronze, still 0.10% short of silver.

### Harvesting the public frontier (2026-10-29): what "more data" actually means

`manwithacat/vqcc-apache-public-circuits` publishes a `manifest.csv` (939 files: 886
circuits, 52 factorizations) that is the map of how every public circuit was made. Each row
carries the source notebook and its version, the licence evidence, the verified local cost,
and, for chemistry, the full metric breakdown. That is the "how they got the data" answer:

- **One shared factorization.** All 52 chemistry bundles are SomeJoint's R8/B143/C60
  full-XOR five-word family. Nobody has a different shape at the frontier.
- **Two independent knobs on it.** (a) *re-gauge* the factorization — an npz-only change that
  moves `W`/`H2_I` and with them λ_eff; (b) *recompile* ANGLE circuits — a circuits-only
  change that moves T. The published chain is
  `d551216fe4b6` (λ 48.764, T 16,366, ours) → Vyas re-gauge (48.689) → Kirill re-gauge
  (48.687) → heliosli rebuild of 5 odd-k batches (48.683, **T 16,296**) → Gluzdov re-gauge +
  rebuild of the 10 even-k batches for its own words (λ 48.632, T 16,306, **340,990,160**).
- **The circuits are interchangeable when the words match.** Swapping one batch's `.kmb`
  between heliosli's 16,296 bundle and Gluzdov's 16,306 bundle still verifies
  (`mix_scan.py`: every single-batch swap PASSes), which means the family shares one word
  set and the per-batch minimum is harvestable. Batches 5 and 7 are 2 Toffolis cheaper in
  heliosli's compilation.

Verified locally, all PASS:

| bundle | λ_eff | T | cost |
|---|---:|---:|---:|
| ours (`d551216fe4b6`) | 48.76383 | 16,366 | 343,169,614.38 |
| heliosli `488a3af26cf3` (submitted, ref 57754329) | 48.68272 | 16,296 | 341,133,443.84 |
| **Gluzdov `0418007f2646`** | **48.63243** | 16,306 | **340,990,159.68** |

Gluzdov alone is composite **22.720803**. This is the honest ceiling of harvesting: −0.63%
on chemistry in total, +0.048 composite.

### The DQI failure penalty, measured exactly

The verifier's own Clopper-Pearson bisection (`dqi_r.py`) at `num_shots = 102,400`,
`log2_delta = -128`:

| failing shots | eps_plus | R | DQI cost | composite |
|---:|---:|---:|---:|---:|
| 0 | 8.6606e-04 | 1.191260 | — | — |
| 1 | 9.1042e-04 | 1.196731 | +0.459% | **−0.153%** |
| 2 | 9.4882e-04 | 1.201388 | +0.850% | −0.283% |
| 3 | 9.8391e-04 | 1.205584 | +1.202% | −0.399% |

So one extra failing shot costs 0.46% of the DQI cost, not 7% — the R formula is
`ln2 / −ln(1/2 + 2√eps)` and eps only moves in the fourth decimal. This is cheap enough that
swapping in a newer DQI file is a bounded bet: `53d8b2edf251` (SomeJoint v19, 897 qubits,
9,810 fewer static Toffolis, `convert`/`unconvert` rebuilt with aligned routing) verifies at
**412,943,252,615.63** locally, −0.44% against `4ee7a17b4cbc`, and it is *not* the 897 file
that drew 2 board failures (`b3354f60e459`). Even at 2 board failures it stays within
0.5% of the incumbent, and at 0 it takes the portfolio to **22.7545**.

### What that leaves for silver and gold

Steady state, at our λ_eff: chemistry 341.13e6 → 340.08e6 (T 16,296 → 16,246, i.e. **50 more
Toffolis**) reaches 22.74105. Or one invsqrt qubit: 63 → 62 is +0.537%, which is 22.86.

But `manwithacat`'s "Is the 63-qubit inverse square root optimal?" closes the 63 → 62 route
far better than we did, and the forum's null list agrees (no borrowable wire, 0 degenerate
controls in ~1,500 Toffolis, dropping any rotation breaks the window). The top ten are 4–15%
above the public plateau, which at the marginal rates above is only reachable with a
**different invsqrt design** — at 26.14 the arithmetic demands an invsqrt ratio near 31, i.e.
~41 qubits at the current per-shot cost, roughly 570 Toffoli-equivalents of slack bought by
22 fewer qubits. That is a design problem, not a tuning problem: it is where the medals are,
and it is not reachable by shaving loader Toffolis.

### What the 22.67 portfolio changed

Only chemistry: somejoint's `qcc-our-chemistry-b143-c60-xor2` (block
`team-b143-c60-xor2`, 13 files, SHA-256 all matched, submission digest `c88f058e…`)
lowers T from 16,528 to 16,366 at the same Q 430 — XOR₂ shares one delta between two
batches. Rebuilt with `_tools/kprobe/run_somejoint_build.py` (exit 127 without a writable
TMP, as below); official verifier PASS, cost 343,172,572.78, ratio 9.7782.

### Re-verified from scratch, 2026-10-29

All three members re-run against the vendored official verifiers via
`_tools/kprobe/run_verifier.py` (DLL dir + writable TMP only; no code changes):

| task | verifier output | matches |
|---|---|---:|
| dfthcsos | PASS, score 343,172,572.78225887, T 16,366, Q 430, λ_eff 48.76425214813735, error₁ 0.01583, error₂ 0.019999, 15 circuits, 78,080 shots, submission `c88f058e…` | 9.7782× |
| dqi | PASS, score 414,781,342,770.7355, T 440,563.955 (max 454,507), R 1.1912604648115832, effective_error_rate 8.6605873e-4, **0 value / 0 phase errors** of 102,400 shots | 58.9213× |
| invsqrt | PASS, score 102,664.23957721348, TE 1,629.5911, toffoli 1,334.5003 (max 1,402), 41 rotations at eps 2.4340415e-13, Q 63, `worst_error_over_tolerance` 0.99932, circuit `07314803…` | 20.2288× |

Composite recomputed as the geometric mean of the three ratios: **22.672535965**,
i.e. the 22.67253 the board gave ref 57739663 — the offline pipeline reproduces the
leaderboard to five decimals, so the local harness is a faithful proxy.

**Cross-build chemistry mixing is impossible (measured 2026-10-29).** The idea was to
take heliosli's cheaper ANGLE circuits for batches 0–3 (303 Toffolis vs our 281+…, Σ4,210
vs Σ4,218 → T 16,350). It fails at the input, not the circuit: `out/dfthcsos/submission.npz`
and `heliosli-live-20261028/dfthcsos__submission.npz` share `H1_I`, `Hsym_lower`, `brot`
and `tree_edges` but differ in **`angle_words`**, `H2_I`, `U` and `W`. A batch circuit
computes the **delta** of the submitted `angle_words`, so a circuit emitted for one
word table decodes garbage under another; and the words come with their own factorization,
whose `λ_eff`/error margins are what the score is computed from. Porting the circuits
therefore means porting the whole npz, at which point it is heliosli's submission, not a
mix. The +0.033% on offer was below the "clear improvement" bar anyway.

### Survey of everything published up to 2026-10-29 — no task has a better circuit

Pulled every notebook and dataset updated since 2026-10-27 and re-derived each reported
number from the published costs. The public frontier is a single portfolio at **22.67266**
(`atakanaldemir/quantum-circuits-xor2-refit-and-gauge`, 2026-10-28 13:26): the same
somejoint XOR2 words with a W/H2_I refit worth −5,900 of 343,172,573 on chemistry, i.e.
**0.0006% of the composite**. Their own ablation is the useful part — CUDA pair gauge
−127, fixed-word refit −4,804, Cayley+pair selection the rest — and it confirms that at
fixed words the chemistry score is a *numerical* problem with a ~0.002% range.

| task | best public | cost | ratio | ours |
|---|---|---:|---:|---:|
| invsqrt | mrmeows 63Q build (all 2026-10-28 notebooks reuse it unchanged) | 103,546.25 | 20.0565× | **20.2288×** |
| dqi | 877Q "narrowed fused decoder" (saddamaditya, ahmedberatozer) | 4.42121e11 | 55.28× | **58.9213×** |
| dfthcsos | rustad R10/B129/C44 `lam47549_canon` (+ mrmeows J=5 lookup) | 3.50414e8 | 9.576× | **9.7782×** |

- The **877-qubit DQI narrowing** buys (877/889)² = 0.9732 on width for +1.74% Toffolis
  (474,308 → 482,543), a net −1.0%, but its T is 6.2% above our 454,507 at 889 qubits, so
  it lands 4.2% worse overall. Register reclamation (early u→z, in-place reverse jet) is a
  real technique; it is being spent to fix a width that our circuit already beats.
- heliosli's dataset is at v11 (2026-10-28 15:57) and its `dqi.kmb.xz` is byte-for-byte
  the 9,144,688 B file we already ship — no new DQI bytes exist.
- invsqrt is closed *independently* of our own measurement: `manwithacat/
  is-the-63-qubit-inverse-square-root-optimal` runs six tests on the shipped circuit and
  every one fails to open headroom — spikes recur at every scale (5.9% of inputs above half
  tolerance, so no cheap region), the spikes are seed-independent, integer rounding in the
  Horner arithmetic fills 59–97% of the tolerance, the best alternative coefficient
  rounding fails by 1.4–3.1× tolerance, and an LP over all 44 rotation angles improves the
  minimum margin by 2 units in 47,884,702 (it is already optimal). A 30× larger margin
  would be worth only 1.75% of the cost.

### The best public factorization recipe, and why it does not transplant

`michaelrustad/vqcc-dfthc-r10-merged-groups` ships the factorizations *and* the training
code, so the public state of the art on the chemistry factorization is documented:

| version | (R,B,C) | λ_eff | T | Q | cost | ratio |
|---|---|---:|---:|---:|---:|---:|
| v1 | 10,129,60 | 48.031 | 26,078 | 447 | 5.599e8 | 5.993× |
| v2 | 10,129,46 | 48.204 | 19,136 | 431 | 3.976e8 | 8.440× |
| v4 | 10,129,46 (λ-descent) | 48.127 | 18,126 | 423 | 3.690e8 | 9.094× |
| v5 | 10,129,44 | **47.549** | 17,139 | 430 | 3.504e8 | 9.576× |
| ours | 8,143,60 | 48.764 | **16,366** | 430 | 3.432e8 | **9.778×** |

Their whole gain is the trade we cannot make: a 2.5% better `lambda_eff` against a 4.7%
worse `T`. The recipe for the λ side is a **D(w) penalty in the quantization loss**
(Λ 295 → 290.6, "with 2e-5 the same schedule ends at 47.89–47.92, above 6e-5 error_2 no
longer fits"), on top of copy truncation with 6000-iteration L-BFGS refits and hybrid
progressive quantization. Their own numbers show the λ lever's size: **1.5% of Λ, i.e.
0.7% of λ_eff, for a multi-hour GPU pipeline.**

Where our `T` comes from is the part nobody else has, and it is the part that cannot move:

- `heuristic_toffoli` is a closed-form function of `(R,B,C)`. Measured with the verifier's
  own `heuristic_costs`: (8,143,60) 7,930 · (10,129,44) 7,511 · (8,143,44) 6,662 ·
  (8,143,30) 5,539 · (6,143,60) 6,741 · (12,143,60) 10,258. So `C` 60 → 44 is worth
  **−1,268** T, and it is the only large knob in the heuristic.
- But `C` is not free to change: any change to `(R,B,C)` or to `U` changes `angle_words`,
  and the 225-product frozen plans are valid only for *these* words. Falling back to the
  shipped `make_plan` costs ~305 Toffolis/circuit against the frozen plan's 281, i.e.
  +1,500–2,400 T — **more than the 1,268 the heuristic would return.** The frozen plans and
  the factorization are a single locked object. (This is why `span_prune.py`/`joint_dict2.py`
  could not reproduce 225: it was not derived from this table by any planner we have.)

### The λ_h2 anatomy on our submission — measured, and its 1.5% ceiling

`probe_lambda.py` on the shipped npz (every number reproduces the verifier bit-for-bit):

```
Lambda 305.6626 = lambda_h1 12.8108 + lambda_h2 292.8518      (h2 = 95.8%)
  budget   0.25 sum L1^2      150.580  (51.4%)     sum L1 462.05 over 480 blocks
           0.25 sum w_B^2      26.465  ( 9.0%)     sum |w_B| 153.70
           0.5  sum L1 |w_B|  115.807  (39.5%)
  L1 / nuclear                 1.1686 mean (min 1.000, max 2.349) -> 0.25 sum nuc^2 139.457
  E_gap identity  E_gap = -536.631 (fixed) + 0.5||W||^2 540.546 = 3.914902
  caps: error_1 15.830/20 mHa, error_2 19.9988/20 mHa, E_gap = floor + 4.2e-6
```

Both gates are saturated and the E_gap is pinned 4.2e-6 above its floor, so the submission
is a *corner* of the feasible set. `w_B` is submitted directly (the verifier derives
`W = w_B - sum_b w`), which makes it look like a free knob at fixed `X` (error_2 and the
two-body words untouched). Measured, it is not: `lambda_spread.py` and
`lambda_gauge_search.py` walk it in 20 directions — `-t`, `+t`, `-y`, `±W`, `±sign`,
`±1`, `±L1`, random, and single-block moves — and every move that buys `|w_B|` pays for it
by *destroying* `h1_prime`: at `H2_I = 0` the negative part of the spectrum goes from
`p_neg 11.25` to `236`, `E_gap` from 3.91 to 174.5, and `lambda_eff` from 48.76 to 292.
The best direction found reaches 48.765 (i.e. nothing). The balancing act that keeps
`p_neg` at 11 while `||W||^2 = 1081` is what the submitted point *is*.

So on chemistry the reachable total is: λ −0.7% (rustad's D(w) descent, if we rebuilt a
PyTorch refit pipeline on GPU) and nothing else. That is +0.23% on the composite.

### The "micro-improvement" path, measured to its ceiling (2026-10-29)

Three levers were tested that keep the circuits byte-identical (so the chemistry verify is
an 18-second run and the submission is risk-free). **Two are exactly zero, and the third is
worth 6e-5 of the composite.**

| lever | what it changes | measured |
|---|---|---|
| orthogonal mixing of the `C` columns (`gauge_lambda.py`) | `λ_h2` only; `X`, `h1_prime`, both errors, `E_SOS`, `E_gap` provably invariant | **0.000%** — all 1,770 Givens pairs in all 8 groups are already at their optimum (λ_h2 292.851798613 → 292.851798613) |
| re-gauging `w` inside the atom map's null space (`nullspace_gauge.py`) | `λ_h2` only, same invariance | **0.000%** — the atom map `w ↦ Σ_b w_b u_b u_b^T` is **injective (rank 143/143 in every group, null space 0)**, so `w` is uniquely determined by `X`; there is no redundancy to spend |
| refit of `w_B` under the verifier's own gates (`wb_refit.py`) | `λ_h2`, `p_neg`, `error_1`; keeps `X`, words, T, Q | **λ_eff 48.764252 → 48.763832 (−0.00086%)**, cost 343,172,572.78 → 343,169,614.38 (−2,958) |

Why the third stops there: `dλ_eff = 0.0803 dΛ + 6.189 dE_gap`, so `E_gap` is 77× more
expensive than `Λ` — every `Λ`-reducing move found also lowers `E_gap`, and `E_gap` may not
go below its **strict** floor 3.914898 (equality is rejected). The feasible point can only be
reached by projecting back onto the floor, and it lands 2.1e-7 above it with `error_1` at
16.78/20 mHa.

That candidate was built and run through the official verifier on 2026-10-29
(`_scratch/chem_gauge/verify_wbrefit/`, 15 unchanged `.kmb` + the refit npz):

```
status PASS   score 343169614.3753365   lambda_eff 48.76383176343135
toffoli 16366   qubits 430   verified_shots 78080
Lambda 305.6577190874813   E_gap 3.9148982126353076   gap_in_domain true
error_1 0.01678059503 (cap 0.020)   error_2 0.01999880580 (cap 0.020)
```

so the model was exact on the final file too. Composite with it:

```
invsqrt 102664.23957721348 -> 20.228799
dqi     414781342770.7355  -> 58.921344
chem    343169614.3753365  ->  9.778254   (was 9.778170)
composite 22.672601117      (was 22.672535965, +0.000065 = +0.00029%)
```

**It is not worth a submission.** +0.000065 composite is one fifth of the distance to the
public frontier (22.67266, which implies a chemistry cost of 343,166,941 — our refit gets
72% of the way there), and it burns one of two daily slots for a number no scoring boundary
can see. The candidate is kept on disk in case a later recipe wants the same corner.

The reduced model is exact, not approximate — it reproduces the official verifier to the
digit on the shipped file: `Lambda 305.662648`, `E_gap 3.914902`, `error_1 15.830 mHa`,
`lambda_eff 48.764252`.

**Ceiling of the whole cheap path: composite ≈ 22.6726–22.6727.** The public frontier
(22.67266, atakanaldemir) is exactly this ceiling — their published ablation is
CUDA pair gauge −127, fixed-word refit −4,804, Cayley+pair selection the remainder, all of
it inside the same corner. Beyond this point the only routes are new circuit designs
(invsqrt/DQI) or a GPU factorization-training pipeline (chemistry), which the sections
above price at +0.23% and +0.33%-per-1% respectively.

### The public frontier, checked on 2026-10-28

- `manwithacat/vqcc-apache-public-circuits` ships a `manifest.csv` of 51 public circuits
  with each one's locally-verified `local_task_score`. Best per task there: invsqrt
  102,664 (07314803…, ours), dqi 4.14781e11 (4ee7a17b…, ours), chemistry 3.47705e8
  (558122c4…, R8/B143/C62, T 16,752) — our xor2 beats every chemistry entry by 1.3%.
- `dmitriigluzdov/vqcc-femoco-public-factorization` now carries `open_circuits_v26`,
  whose dqi is byte-identical in size to heliosli's 889-qubit circuit (9,144,688 B xz) and
  whose chemistry is the same R8/B143/C60 family. His v21–v25 DQI circuits were all
  worse (13.9 MB → never adopted by him).
- `kirilltym/quantum-circuits-exact-gauge-research-assets` claims 344,812,530.9 chemistry
  from a pair-gauge search on the V16 archive — 0.5% worse than xor2, and his dqi/invsqrt
  are ours.
- Our own banked-QROM (`four_addr`) cannot beat their emitter on this shape: at
  (J=5, s=1, brot=8) it fits in 91 qubits but costs 693 Toffolis per circuit against their
  281 (10,395 vs 4,218 per pass). `banked_qrom.build_circuit` is now parameterised on
  `brot` (bank width `brot-1`) so this is testable, but the construction loses here.

### Three negative results on public-code optimisation (2026-10-28)

Each was measured, so none of them is worth retrying as-is.

1. **Re-planning the ANGLE circuits with their own shipped planner.** `paired_anf.make_plan`
   plus `scheduler.make_schedule` are in the Apache-2.0 block. Over
   mixing_steps 0–3 × share_odd × in_place_parity × seeds, the best it reaches is 263
   products for the early batches and 235/230 for the late ones, against the frozen
   plans' 225/180 (T = high + low_computations + products: 69 + 11 + 225 = 305, which
   matches `BUILD.json` exactly). Their planner does strictly more than the shipped code.
2. **Reducing `lambda_eff` by the orthogonal gauge.** The physics only sees
   `sum_c w_bc u_b u_b^T` and `sum_c W_c X_c`, so `(w, H2_I) -> (w O, H2_I O)` leaves X,
   h1_prime, both errors, E_SOS and E_gap untouched while moving the LCU one-norm
   `lambda_h2 = 1/4 sum_c (sum_b |w_bc| + |w_Bc|)^2` (`lambda_eff ≈ sqrt(2 E_gap Λ)`).
   The Frobenius floor is `0.25*||V||_F^2 = 277.23` against the current 292.85, i.e. at
   most Lambda 305.66 -> 290.04 and lambda_eff 48.7643 -> 47.4936 (cost −2.6%, composite
   +0.88%) — but that floor needs every column to become a singleton. Measured: a full
   Givens pair sweep over all 1,770 column pairs per group moves the objective by
   **exactly 0.00%**, and 200 random Haar gauges plus 200 small Cayley perturbations per
   group are all worse (e.g. rho 0: 147.26 -> 308.76 under a random rotation). The stored
   weights are already at a strong local minimum, which is consistent with avazaku's and
   kirilltym's public attempts both failing to beat xor2.
3. **Their schedule with a smaller low cache.** `make_schedule` only succeeds at
   capacity 11 (the low cache has 11 masks); their frozen schedules use capacity 6/8 with
   the same 11 low computations, so their scheduler is also richer than the shipped one.
4. **A long GL(m,2) monotone search over the output basis.** `mix_output_basis` flips two
   columns with `funcs[i] ^= funcs[j]`, `fanouts[j] ^= fanouts[i]` and accepts only a strict
   cost decrease; `make_plan` calls it with steps = mixing_steps, whose default is 0, so the
   shipped planner effectively never runs it. Cost is `ceil(rank/2)` products per column and
   `rank_decomposition` is already exact Gaussian elimination, so the basis is the only
   degree of freedom. Measured: 6,000 random proposals × 2 restarts per batch accept
   **nothing** — every T stays at the base value (280/268/232 products, i.e. T 360/348/312
   against the frozen 305/283/260), and the searched total is 5,100 per pass vs the frozen
   4,218 (+20.9%). Even at the base basis the search confirms `share_odd` is the only budget
   lever (280 -> 263) and it still lands short of 225.
5. **Annealing the same basis (exact moves, non-monotone).** 40,000 proposals on batch 0,
   temperature 1.2 -> 0.02, with every state verified exact on all 1,220 addresses: the best
   reached is the starting 280 products. Reason: 280 = 40 columns × 7 = `ceil(rank/2)` with
   rank 13–14 in every column, i.e. the base basis already sits on the per-column floor;
   their 225 needs an average of 5.6, so columns with rank <= 12, which no combination of
   these polynomials reaches.
6. **Merging products that share a factor pair.** Their per-output-bit membership is ~4,100
   over 225 products (18 bits each) against our ~2,900 over 263 (11 each), which looked like
   fanout merging. Measured: **0 collisions** — no two columns in any of the sampled batches
   share a `(left1, right1, left2, right2)` key. It also turned out to be the wrong question;
   see the corrected cost model below.
7. **Choosing a different set of four folded address bits (free under the metric, since CX
   is not billed).** Blocked by arithmetic: A = 1220 is not a power of two, and the top
   address bit cannot be permuted with any other without leaving the valid address set, so
   the identity is the only legal split.
8. **Re-coding those four folded bits by an invertible GF(2) map.** The last partial 16-block
   (addresses 1216–1219) stays valid iff the map preserves `{0,1,2,3}` as a set, which admits
   576 of the 20,160 GL(4,2) matrices. All 576 were scored on batch 0
   (`_tools/kprobe/chem_lowcode.py`): **every one gives exactly products 280 / high 69 /
   low 11, T 360**. This is structural, not luck — a linear change of variables induces an
   invertible (degree-blocked) transform on the 16 low monomials, i.e. a row operation on the
   16×77 matrix, so the rank and hence the cost is invariant.
9. **Merging blocks with equal polynomial effect.** The two terms of a product are symmetric,
   `(L1^R2)(L2^R1) = (L2^R1)(L1^R2)`, so different factor quadruples can be the same
   polynomial, and equal-effect blocks with disjoint fanouts merge into one billed CCX.
   Measured on batches 0 and 5: **225 and 180 distinct effects for 225 and 180 blocks — zero
   mergeable pairs.** Their plans are tight against this edit too.
10. **A joint dictionary greedy over monomial-pair blocks.** See the corrected cost model
    below for why this is the right framing. Batch 0: 40 blocks buy only ~55 support
    each out of 24,242, so the loop needs ~440 blocks — roughly twice their 225.
11. **A joint dictionary greedy over rectangular blocks with a rank objective.** Candidates
    are the rank-2 factor pairs of the densest residual, scored by total rank removed across
    all 40 columns (`_tools/kprobe/joint_dict2.py`). Sharing does not emerge: almost every
    block serves exactly one column (drop 2, 1 column), so the run projects to ~300 blocks
    against their 225. The reason is visible in the numbers — a block drawn from one column's
    own decomposition is bespoke to that column, whereas theirs are shared ~18 ways.
12. **The `{16 low monomials} × {~14 high masks}` product-dictionary hypothesis.** 225 ≈ 16 × 14
    and their blocks are single-low-monomial × ~37-high, so a shared row space looked likely.
    Disproved: the 640 row vectors (40 columns × 16 low monomials, as vectors in GF(2)^77)
    have **GF(2) rank 77** on batches 0, 4 and 5. The high side is full-dimensional, so no
    small generator set exists for it and 14 is not a subspace dimension.
13. **Pushing the boundary rows into the free `pure` channel.** `Plan.toffoli` bills only
    cache + products, and `pure` entries are emitted as free CX. Every boundary slot
    (`h = 0`, indices 0–15, and `l = 0`, indices 16·h) is itself a monomial, so the whole
    boundary can be corrected for free and only the 1,128 interior slots need products.
    Measured: interior rank equals full rank in **every** column of batches 0, 4 and 5
    (all 15, or 4 where a column is already rank 4), so the boundary contributes no rank
    and this is worth **0.00%**.

### The size of the remaining gap

Per-column floors on the mixed basis vs their frozen plans: 280 vs 225, 238 vs 180,
268 vs 203 — a consistent factor of ~1.29–1.32. Summed over the 15 plans the frozen
dictionary is 3,018 products against our planner's best ~3,900, so beating them needs a
shared-dictionary construction at least 23% better than the per-column floor, under a
metric that bills only dictionary size. Every construction tried above lands at 300+ per
batch instead.



### The corrected cost model for the ANGLE stages (2026-10-28)

This overturns the reasoning behind items 4–6 and matters for any future attempt.

`Plan.toffoli = len(cache_masks) + len(products)`, and a product's `fanout` is a 40-bit mask
over the output bits: `build_circuit` emits `bits(prod.fanout)` as CX, and **the metric does
not bill CX**. So the billed quantity is the number of *distinct blocks*, and a column may
consume any number of them for free. A product

    (L1(lo) ^ R2(hi)) & (L2(lo) ^ R1(hi))
      = L1L2(lo) ^ L1(lo)R1(hi) ^ L2(lo)R2(hi) ^ R1R2(hi)

is a block: two cross monomials *plus two free pure terms*, XORed into every column in the
fanout. The plan is therefore a shared feature dictionary, not a per-column decomposition.

Measured against their frozen batch-0 plan (decoded by `_tools/kprobe/decode_plan.py`):

| | theirs | our `make_plan` |
|---|---|---|
| products | 225 | 280 |
| columns | 40 | 40 (mixed basis) |
| column ranks | 15, all 40 | 13–14 |
| per-column floor `sum(ceil(rank/2))` | 320 | 280 |
| (column, block) incidences | 4,087 | 280 |
| columns served per block | 18.2 | 1.0 |

Their plan is *not* minimal per column — every column is rank 15 (a floor of 320) and each
receives ~102 blocks, yet only 225 blocks are billed because each is shared. Block shapes are
a single low monomial × ~37 high monomials plus a small second term, mean effect 1,556
monomials; the toggles cancel heavily (350,108 monomial-toggles for 40 × 1,220 = 48,800
slots). All 225 factor quadruples are distinct, so this is not a product structure
(46 distinct `left1` values) and not a dedupe of their own decompositions.

The consequence: **225 is not reachable from the shipped per-column planner**, and the
shipped code cannot reproduce it either (it gets 280). It came from
`frozen_from exp/onebody_loader_port/five2_a16`, an experiment that is not in the
Apache-2.0 tree we downloaded, so the construction is unavailable. Since we already ship
their frozen plans, beating them means rediscovering a joint feature-sharing construction
that their own published code does not contain — the block counts above are the evidence
that both natural greedy formulations miss it.

Roofline, corrected. Their T 16,366 splits as 2 × 4,218 staged (forward + reverse over the
15 batch plans, `per_pass_T` and `audit.actual_forward_reverse_T` = 2 × 305 on batch 0) plus
7,930 for the heuristic stage, which no plan edit touches. So dT = 2 × Σ_batch d(products),
and the 15 plans bill 4×225 + 5×180 + 6×203 = 3,018 products. The chemistry term needs cost
× 1/1.0302 for 22.9, i.e. T 16,366 → 15,886, i.e. Σ products 3,018 → 2,778: **240 fewer
products across the 15 plans, 16 per batch on average (−8%)**. Still a much smaller step than
the previously recorded "265 Toffolis/circuit" target, which wrongly assumed 60 circuits at
the early-batch 305 — but no route to even that 8% is currently known.

### The frozen plans are irredundant, so pruning cannot help (measured, all 15 batches)

`_tools/kprobe/span_prune.py` evaluates each product's predicate on all 1,220 addresses (so
the block effects match `Plan.evaluate` exactly), takes each column's interior straight from
the frozen fanouts (`pure` never touches the interior), and measures the spans:

- **dim W_int = 35** on every batch (the 40 columns' interiors are not independent).
- **dim span(blocks) = #blocks exactly**: 225/180/203, never less.

So the map from block coefficients to interior is injective and every column's subset
representation is *unique*. A block cannot be dropped or re-derived from the others, which is
why the greedy cover keeps all 225: there is nothing to prune. It also rules out the whole
"the plan is a redundant code, so a spanning subset is cheaper" family of ideas, and it means
the only way to a smaller dictionary is a genuinely different one (the floor is 35 blocks, but
35 blocks would have to *be* elements of W_int, and a 35-dim subspace of the 1,128-dim interior
essentially cannot contain a rank-2 element).

The rest of the chemistry term is not ours to move either. `heuristic_costs` is
`SpinFreeData_V1(R=8, B=143, C=60, N=76, chi=KEEP_BITS, beta=8)` — every argument comes from
the Hamiltonian, including `beta = brot`, so the 7,930 Toffolis are fixed by the physics; and
`Q_total` = heuristic persistent + `max(heuristic temporary, angles temporary)`, i.e. dominated
by the heuristic workspace (430 against the ANGLE plan's own 126). Both errors are also
saturated. Only the 2 × 4,218 staged Toffolis are ours.

### The other two terms are closed too (measured 2026-10-28)

**dqi.** `score = repetition_factor × Q² × mean_toffolis`, and the repetition factor is a
*statistical* floor, not a circuit property: solving `R = ln2 / -ln(0.5 + 2 sqrt(eps))` for our
R = 1.1912604648115832 gives eps = 8.66e-4, which is exactly
`1 - 2^(-128/102400)` = the Clopper-Pearson bound for **zero errors in num_shots = 102,400 at
log2_delta = -128**. Our circuit already made zero errors, so R is at its floor and no circuit
can beat it. Q = 889 is 49 above the hard floor (70 syndromes × 12 bits = 840 input qubits
must be held), and the reference builder needs 1,893 qubits / 5,725,461 Toffolis against our
889 / 454,507 — heliosli's rebuild is ~57× cheaper on Q²·T. Nothing left that we can reach.

**invsqrt.** `score = Q × TE`, `TE = toffoli_count + t_count/4`, and the rotation term is
`t_count = t_base + n_rot·(0.56·log2(1/eps) + 5.3)` with
`eps = π·min_margin/2^64/n_rot` — the verifier always spends *all* the leftover tolerance, so
eps is not a knob. Our numbers (41 rotations, eps 2.434e-13, 28.77 T each, t_count 1,180.36)
fit that exactly, and `worst_error_over_tolerance = 0.99932`: the tolerance is 99.93% consumed,
so there is no margin to buy back. The rotation total is `1 + n(0.56 log2 n + 25.77)`, which is
log-flat (41 → 20 rotations is only −9.4% of TE), and n_rot is the algorithm, not a free
parameter. Q = 63 and toffoli_count = 1,334.5 come from an already re-synthesised public
circuit (172 ANDs, down from 186).

So all three terms sit on floors that come from the task's own cost models or from optimised
public artifacts, and the composite stays at 22.6725.






### How the 22.63 portfolio was assembled

- **invsqrt** — `busyaprime/33-qcc-v8-members` v4, Apache-2.0. mrmeows' 63-qubit circuit
  with the main lookup re-synthesised from 186 to 172 AND gates (−0.85% cost).
- **dqi** — `heliosli/qcc-circuits-live`, Apache-2.0: `dqi.kmb.xz`, 889 qubits, unpacked
  SHA-256 `4ee7a17b…`. T 440,564 against v9's 482,543, which more than pays for the 12
  extra qubits (Q² up 2.8%, T down 8.7%).
- **dfthcsos** — somejoint's R8/B143/C60 full-XOR design, rebuilt locally from the frozen
  plans in `somejoint/qcc-our-chemistry-b143-c60-xor` plus the source embedded in their
  notebook. The build is self-checking: every circuit's SHA-256 must equal the frozen one,
  each circuit is run as an XOR on random words across all 1,220 addresses forward and
  back, and the submission digest must equal `07dfbd32f101…`, the one the verifier passed.
  It reproduced exactly. Building it needs `TMP`/`TEMP` pointed at a writable directory —
  kickmix's `to_file` goes through a C `tmpfile()` that fails on this machine's default
  temp (the same failure `env.py` documents); without that the process dies with exit 127
  and no traceback.

Licensing: all three datasets are Apache-2.0 and keep their full credit chains. The DQI
lineage still runs through jday96314's unlicensed QuantumAI repository; heliosli ships only
the compiled circuit (not the code) and asks that the author be contacted before reusing
the code.

## The v9 portfolio (previous best, ref 57734440)

Composite 21.8462 locally, 21.87960 on the leaderboard — rank 536/1398 (top 38.3%), where
the top-10% cut was 22.5736 and #1 was 26.1411. The 0.15% gap between local and board is
the dqi repetition factor's shot sample.

Verified per component:

- **invsqrt** — `verify_invsqrt_ext.py`: score 103,546.168164559, TE 1,643.59, toffoli
  1,348.499, 63 qubits, 41 rotations, worst_error/tol 0.99932.
- **dqi** — unpacked SHA-256 `f3d335c3…` matches the published digest. `verify_dqi_ext.py`:
  PASS, score 444,151,956,391.49, T 482,542.61, R 1.19673, 877 qubits, 114,178 bits,
  8 errors in 102,400 shots. README quotes R 1.1913 (a different verifier salt), so the
  cost here is 0.5% above its 4.4212e11 — same circuit, different shot sample.
- **dfthcsos** — `shipped_breakdown.py`: PASS, T 17,330, Q 431, 128 ANGLE qubits
  (11 selection + 40 persistent + 77 temporary), error₁ 14.658 mHa, error₂ 19.998 mHa,
  `gap_in_domain` true, E_gap 3.915093 (floor 3.914898, margin 1.9e-4), λ_eff 47.55334.
  15 ANGLE circuits, batch 5, 87,424 shots.

Note: the verifiers must run under `solutions/vqcc/.venv/Scripts/python.exe` (3.14.6);
`build/kickmix.pyd` links `python314.dll`, so the system 3.12 cannot import it.

### Why our own levers could not close the gap (superseded: the public 22.63 portfolio did)

The top-10% line on 2026-10-28 is 22.5736 (rank 139 of 1,398). From v9's 21.8796 the
composite had to rise 3.17%, which means the **product of the three ratios had to rise
10.3%**. Nothing we could build ourselves reached that; what did reach it was adopting
three public Apache-2.0 circuits that are each a little ahead of v9. Measured, not guessed:

- **Chemistry shapes.** `cost_scan_v9.py` sweeps (R,B,C) through the verifier's own
  `SpinFreeData_V1` at N=76, brot=8. Holding the ANGLE term at the v9 measurement
  (9,624 Toffolis) and λ_eff at 47.553, the cheapest shape in the grid is (10,129,30) at
  +10.9%, but it has R·C = 300 against v9's 460 — and R 10→9 alone (R·C = 414) already
  measures 22.1 mHa, over the 20 mHa gate. Restricting to **equal capacity** (R·C 440–520)
  the best shape is (8,129,55) at only −1.6% cost, and it needs R to drop to 8, i.e. two
  merged groups, which is worse than the measured one-group merge. So chemistry's
  reachable headroom is ~1.6%, worth +0.5% on the composite.
- **brot.** 7-bit would take Q from 431 to 393 and T by 300, i.e. −10% cost, exactly what
  is needed — but measured error₂ at brot 7 is 349.7 mHa, and quantization-aware retraining
  at three shapes still sat at 30–40 mHa. 8 bits is forced. 9-bit costs more Q than any
  shape saving it enables.
- **DQI.** Cost is Q²·T·R. Q = 877 is the common floor of the 12-bit inverse and scaling
  kernels; Alex measures the exchange rate at ~1,100 Toffolis per qubit and the cheapest
  narrowing available at 23–42 Toffolis **per window across 2,046 windows**, so narrowing
  loses. R is a function of the observed failure count, worth only 0.05–0.16% between
  salts. T = 482,543 is 1.7% above the 889-qubit build and the reverse jet is already the
  cheaper of the routes measured.
- **invsqrt.** Cost is TE × Q = 1,643.59 × 63. 63 qubits is closed publicly; the rotation
  budget is saturated everywhere (dropping 1 rotation gives 1.354× tolerance), and the
  reachable-x finding changes cost by 0.08%.

Conclusion: we sit at the public frontier (the public plateau is 21.2–21.5 and v9's 21.88
is the best public portfolio). A further 3.17% is not reachable by any lever measured here;
it needs a structural change on one task.


`ahmedberatozer/vqcc-v9-components` is published under the license field
**"Other (specified in description)"**, not Apache-2.0. The dataset README states "Our own
parts are released under Apache-2.0", which covers the narrowing, the re-quantization and
the J5 compiler, but the upstream chain is mixed:

- **invsqrt** — mrmeows `qcc-1375-open-circuits` v3, Apache-2.0 (clean).
- **dqi** — SomeJoint/Agniv Banerjee's notebook (source embedded there) → alex chilton's
  883-qubit Schur decoder → **jday96314's GitHub repo, which states no license**. The v9
  notice says so explicitly. This is the one link in the chain without a permission
  grant; the competition requires winning submissions to be open-sourced under
  Apache-2.0, which we cannot unilaterally do for jday96314's code.
- **dfthcsos** — Michael Rustad's (R,B,C) factorization plus mrmeows' compiler
  foundations; both used as public competition material.

### The previously shipped 6.7267 point

| task | cost | ratio |
|---|---:|---:|
| invsqrt | 157,841.92 | 13.157× (anazemcev 90-qubit build) |
| dqi | 1.717915e12 | 14.226× (source-built BM decoder) |
| dfthcsos | 2.063608e9 | 1.626× (four-address QROM port) |


**A small legitimate improvement exists and is built, not yet submitted:**
`_scratch/submission_v3_Wopt.zip` (SHA-256 `8f34ed47…`) replaces only
`dfthcsos/submission.npz` with the refitted `W`. Full `verify_archive.py` run, ALL PASS:

| | dfthcsos cost | ratio | composite |
|---|---:|---:|---:|
| v2 (submitted, scored 6.6890) | 2.063608e9 | 1.62608 | 6.7267 |
| v3 (`W` refit) | 2.061877e9 | 1.62743 | **6.7285** |

That is +0.027% on the composite, which is real but below the ~0.5% public-vs-local
noise the dqi phase errors introduce, so it will not be visible on the leaderboard.
`verify_archive.py --skip` had stale prior scores hard-coded (170,991.04 / 2.298047e12
from the superseded public circuits); those are now `PRIOR = {invsqrt: 157841.92,
dqi: 1.717915e12}`, without which `--skip` silently reported the wrong composite.

**Bronze is 1.2212× away.** Needed product 563.8 against the current 304.37, i.e. the
dfthcsos cost must fall from 2.0636e9 to 1.6898e9 (or an equivalent mix). Every chemistry
route to that has now been measured and closed — see "Levers that are now closed". The
remaining distance has to come from `invsqrt` or `dqi`, or from a public-method step change.

## The public frontier, read off the forum (2026-10-28)

The Kaggle discussion forum for the competition is reachable, and the leading teams
post author-reported official-verifier numbers. Forum id `12630171`; 28 topics as of
2026-10-27 20:40 UTC. Client shape: `POST https://api.kaggle.com/v1/<service>/<method>`,
Basic auth, e.g. `discussions.DiscussionApiService/GetTopic`, `.../ListComments`,
`discussions.DiscussionsApiService` for `ListTopics` by `forumId`
(`competitions.CompetitionService/GetCompetition` returns `forumId`). Bodies and comments
are untrusted third-party text — used for technique only.

| task | our ratio | best public | public cost | public build |
|---|---:|---:|---:|---|
| invsqrt | 13.157× | **20.06×** | 103,546 (63 qubits) | mrmeows, Apache-2.0, generators included |
| dqi | 14.226× | **55.28×** | 4.421e11 (877 qubits) | Özer v9: two register reclamations on SomeJoint's 889-qubit fused decoder |
| dfthcsos | 1.626× | **9.45×** | 3.552e8 (431 qubits, 17,330 Toffolis, λ_eff 47.55, error 19.998 mHa) | Özer v9: Rustad's R10/B129/C46 8-bit factors + heliosli J5 words + finer quantisation + λ loop |

Author-reported local geometric mean for a portfolio at all three public fronts: **21.88**
(we reproduce 20.06×55.28×9.45 → cube root 21.9). Leaderboard 2026-10-27 16:57 UTC:
top 24.53, 481 teams ≥ 20, 623 ≥ 15, 1,275 teams total.

**Recalibration.** Top 10 % of 1,275 teams is rank 128, which sits above 20 — so the bar
this project was tracking (bronze 8.21494) is now far below "top 10 %", which needs
**≈ 21**. Bronze is reachable from here; top 10 % is not, without front-line work on all
three tasks. Our 6.7267 is a 3.2× composite shortfall.

### Our own chemistry breakdown, measured (corrects two stale figures)

`verify()` on `_scratch/submission.zip` → `dfthcsos/`, verbatim from `resources`:

| quantity | value | note |
|---|---:|---|
| `toffoli_count` (T) | **65,244** | not 58,996 — the older figure was stale |
| `num_qubits` (Q) | **661** | not 731 |
| `angles_toffoli` | **56,450** | **86.5 % of T** |
| `heuristic_toffoli` | 8,794 | outer 627 + inner 5,317 + givens 2,850 |
| `num_circuits` / `batch_size` | 25 / 3 | per-circuit 2,258 = **1,129 Toffolis per pass** |
| `angles_qubits` | **128** | `selection` 12 + `persistent` 30 + `temporary` 86 — **at the 128 cap** |

**The chemistry gap is almost entirely the ANGLE construction.** The frontier's
per-circuit number is 321 Toffolis per pass (consistent with the forum's "his 321 Toffolis
per circuit sit on the 262-product rank bound plus the caches"; 15 circuits × 642 + 7,706
heuristic = 17,330 exactly). Ours is 1,129 per pass: **3.5× worse on 86.5 % of T**.

Reproducing the frontier from the official cost model (`heuristic_costs(76, 10, 129, 46, 8)`
= 7,706 T / 391 heuristic qubits, plus the ANGLE family at 321/pass, J=5):

| | T | Q | cost | ratio | composite |
|---|---:|---:|---:|---:|---:|
| shipped (65/60/30, b10, ours 1,129/pass) | 65,244 | 661 | 2.064e9 | 1.63× | 6.7267 |
| frontier shape, **our** ANGLE cost | 57,908 | 415 | 1.150e9 | 2.92× | 8.11 |
| frontier shape + J5 + 321/pass | **17,330** | **431** | **3.55e8** | **9.45×** | **12.09** |

The middle row is the key one: the factorization shape alone is worth ×1.80 composite even
holding our circuits fixed, and it is unconditional. The rest is the circuit.

### Reaching 321 Toffolis per pass: it is `A = N + R·B`, not the algebra

The ANGLE family is our own construction (`code/four_addr/banked_qrom.py`, an s-bit
banked QROM re-implemented from scratch), and `design_sweep.py` already carries a
calibration measured on the shipped build:

    per_circuit ≈ H + 9·J·(2^s − 1) + cleanup(n),      H = A / 2^s,   A = N + R·B

At the shipped point (s=2, J=3, A=3976, H=994) that gives 1,129 Toffolis per pass, which
is what the official verifier counts. **So per-circuit cost is linear in `A = N + R·B`.**

| | A = N + R·B | H (s=2) | measured/expected per pass |
|---|---:|---:|---:|
| shipped 65/60/30 | 3,976 | 994 | **1,129** (measured) |
| frontier 10/129/46 | 1,366 | 341 | ~390–450 |
| frontier 10/129/60 | 1,846 | 461 | ~510 |

The frontier's 321/pass is therefore the same construction at a 2.9× smaller address
space, not a cleverer circuit. Checking the tables for an exploitable shortcut came up
empty anyway: on a 32×32 (rho, b) sub-block with no padding, the 750 output booleans
(75 angles × 10 bits) have a median of **480 monomials of degree ≥ 2 out of 1,023**, and
**zero** of them are shared across booleans — i.e. ANF-random, the same conclusion Alex
Chilton reports for Özer v9 ("full GF(2) rank 15, exactly like a random matrix").

Consequence: **every chemistry lever is the same lever — shrink the dictionary.** R·B
drives the ANGLE Toffoli count linearly; R·C drives `outer`/`inner` state prep; and the
8-bit word is what halves `temporary_qubits`. A single refit to the frontier shape
(10, 129, 46) at 8 bits buys the whole 5.8×, with no circuit-design work.

### Next step, and the one piece missing to run it

The experiment is a single refit to the frontier shape. `train_rank.py R B C EPOCHS` already
drives the official trainer at any shape, and `load_seed_params` warm-starts a smaller rank
from the (65, 60, 60) baseline by copying the first `min(R,65)` rows — so

    python train_rank.py 10 129 46 20000 --out ../_scratch/chem_r10_b129_c46.npz

is runnable today. What is **not** wired yet: scoring that fit. `train_small.py` scores each
checkpoint with `official_score(params, tree, brot)` against the *shipped* `angle_words`
table, which has shape (3976, 75) and belongs to the (65, 60, 30) factorization, so it
cannot score a (10, 129, 46) fit. A new shape needs its angle words re-encoded from the new
`U`/`W` (the official `starter/task2_angles/encode.py` path) before error_2 means anything.
`train_small.py` also hardcodes `rank=65` and ignores its own `ref_b`/`ref_c` — misleading,
not wrong, but it means the flag alone is not the fix.

Order of work: (1) re-encode angle words for an arbitrary `(R, B, C)` shape from a
factorization; (2) score a checkpoint with the official numpy path; (3) only then train.
Passing the 20 mHa gate at (10, 129, 46) is worth composite 6.7267 → 12.09 on its own.

### The pipeline now exists end to end, and what it projects

Two pieces were missing. Both are now in place and validated:

* `code/chem_score.py` — scores any `(R, B, C)` factorization with the **official Part-2
  encoder** (`starter/task2_angles/encode.py:encode_inputs`, two-body words → decoded `X`
  → one-body eigenvectors → re-encoded on the same tree) and the official metrics, for
  both tree families and any `brot`. Validated by reproducing the shipped build exactly:
  `log_depth`, brot=10 → error_1 13.330 mHa, error_2 19.980 mHa, Lambda 293.5256,
  E_gap 3.9265, lambda_eff 47.8504 — identical to `verify()`. It also reproduces the
  known 8-bit null on our own data (`log_depth` brot=8 → 56.3 / 181.9 mHa, fail).
* `code/train_rank.py` — now saves its result as a verifier-format `.npz` instead of only
  returning it, so a run can be scored afterwards.

Training at the frontier shape is **cheap**: the official trainer accepts
`(10, 129, 46)` warm-started from the `(65, 60, 60)` baseline (`load_seed_params` copies
the first `min(R,65)` rows), at 12 ms/it early and ~111 ms/it at epoch 190, i.e. about
1.5 h for 50,000 epochs on this CPU. The 200-epoch probe was still at error_2 517 mHa and
E_gap 91 Ha, so the shape needs a real run — capacity is plausible (`R·C = 460` sits above
the Eckart–Young floor of 290) but unproven.

`code/proj_r10.py` turns the calibrated ANGLE model into a score projection. It reproduces
the shipped point (1.62× vs 1.626× measured), which is the check that it is honest:

| shape | brot | J | A | per-circuit | T | Q | ratio | composite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped 65/60/30 | 10 | 3 | 3,976 | 1,143 | 65,944 | 661 | 1.62× | 6.71 |
| frontier 10/129/46 | **8** | 3 | 1,366 | 460 | 30,681 | 415 | **5.54×** | **10.12** |
| frontier 10/129/46 | 10 | 3 | 1,366 | 460 | 29,745 | 724 | 3.27× | 8.49 |
| frontier 10/129/60 | 8 | 3 | 1,366 | 460 | 32,047 | 415 | 5.30× | 9.97 |
| frontier + their ANGLE (321/pass, J=5) | 8 | 5 | 1,366 | 321 | 17,336 | 431 | 9.45× | 12.09 |

So the realistic ceiling of this path, with **our** circuit: **5.54× chemistry → composite
10.12** (above bronze 8.21494, roughly rank ~600 of 1,275). Reaching 12.09 additionally
needs the frontier's J=5 ANGLE construction, which does not fit our banked-QROM qubit
budget (`s=3` is refused by the 128 cap in the table above). The 8-bit word is worth
8.49 → 10.12 all by itself, which is why the refit has to be quantisation-aware.

### Measured: the (10, 129, 46) refit by warm-start truncation fails (2026-10-28)

`train_rank.py 10 129 46 20000` — 20,000 epochs, 45 min on this CPU, warm-started from the
(65, 60, 60) baseline (the first 10 of 65 rows, nothing for the 69 new bases). Scored with
`chem_score.py` (official encoder), not the trainer's internal objective:

| tree | brot | error_1 | error_2 | Lambda | E_gap | gate |
|---|---:|---:|---:|---:|---:|---|
| log_depth | 10 | 11.827 | **159.602** | 681.427 | 4.2788 | fail |
| log_depth | 8 | 51.184 | 279.363 | 683.591 | 6.4431 | fail |
| path | 10 | 33.730 | 214.117 | 682.324 | 5.1761 | fail |

Two independent failures, either one fatal:

* **error_2 plateaus at 8× the gate.** The official number is 159.6 mHa against 20. The
  trainer's own printout ends at "ε_2 = 672.7 mHa", so the trainer's ε_2 is *not* the
  verifier's error_2 — watching it during training is misleading (it is 4.2× pessimistic
  here). Do not tune against it.
* **Lambda is 2.3× worse**, 681.4 against our shipped 293.5, so `lambda_eff` would be
  `sqrt(4.2788·(2·681.43 − 4.2788)) = 76.25` against 47.85. Even if the accuracy gate
  passed, that 1.59× would eat most of the A-shrink: cost 76.25 · 30,681 · 415 = 9.71e8,
  ratio 3.45×, composite **8.68** — not the projected 10.12.

The trajectory (error_2 268 → 517 → 733 → 672 on the trainer's scale, then 159.6 official)
is a plateau, not slow progress: 16,000 epochs moved the official quantity by very little
after the first third. `load_seed_params` truncates 65 rho rows to 10 and leaves the 69 new
bases at their initialisation, which is a poor basin; the public (10, 129, 46) factors were
"quantisation-aware training runs on GPUs with the starter-kit trainer", i.e. a *training
procedure*, not a shape you can warm-start into. This matches the published nulls at this
size: `R = 1` stalls at 20.5–22.4 mHa, `B = 255` stalls at 48 mHa.

**So the diagnosis stands but the route does not.** Shrinking `A` is still the whole lever
(the ANGLE model is unchanged), but reaching a passing factorization at that shape is a
training problem we have not solved.

Annealing stage 1, `(65, 129, 46)` (20,000 epochs, 45 min, warmed from the 50k-epoch
baseline with only B/C truncated):

| tree | brot | error_1 | error_2 | Lambda | E_gap | gate |
|---|---:|---:|---:|---:|---:|---|
| log_depth | 10 | 12.534 | **40.566** | 337.226 | 5.8933 | fail (error_2) |
| path | 10 | 40.551 | 92.211 | 337.405 | 6.0724 | fail |

Better than the direct cut to R=10 (159.6 mHa), so the ladder direction is real — but at the
*easiest* rung we are still 2× over the accuracy gate, and `Lambda` is 337 against the
shipped 293.5 (so `lambda_eff` 60.3 against 47.85). Note `A = 76 + 65·129 = 8,461` at this
rung, so it is unusable as a submission even if it passes — it only matters as a seed.
Holding error_2 while cutting R from 65 to 10 is the whole difficulty, and the rung where
it is easiest already fails.

## Where the invsqrt and dqi gaps actually are (2026-10-28)

Both cost formulas were confirmed against **two independent points each** — our verified
run and the forum's author-reported frontier — before being used as a lever model
(`code/gap_decomp.py` reproduces every number below).

**invsqrt: `cost = toffoli_equivalent × num_qubits`.** Confirmed exactly:
`1753.7991 × 90 = 157,841.92` = our verified score; the frontier's `103,546 / 63 = 1643.5873`.

| | ours | frontier | factor |
|---|---:|---:|---:|
| toffoli_equivalent | 1,753.80 | 1,643.59 | 1.067× |
| num_qubits | **90** | **63** | **1.429×** |
| cost | 157,841.92 (13.157×) | 103,546 (20.06×) | 1.524× |

Qubits are **84.6 %** of the gap. Our rotation/Toffoli content is only 6.7 % worse; we are
doing essentially the same arithmetic with 27 more wires. `90 → 78` alone gives ratio 15.18×
(composite 7.2); `90 → 63` gives 18.80× (composite 7.58).

**dqi: `cost = Q² × T × repetition_factor`** (our own `peak_report.py` docstring says
`R·Q²·T`, where its `R` is the repetition factor, not the reported `r = 2016`). Confirmed
exactly: `1789² × 445,229 × 1.205584 = 1.7179e12` = our verified score, and the frontier's
implied `R = 4.421e11 / (877² × 482,543) = 1.1912`.

| | ours | frontier | factor |
|---|---:|---:|---:|
| Q² | 1789² | **877²** | **4.161×** |
| T | 445,229 | 482,543 | **0.923×** (we are better) |
| repetition factor | 1.2056 | 1.1912 | 1.012× |
| cost | 1.7179e12 (14.226×) | 4.421e11 (55.28×) | 3.886× |

Qubits are **105 %** of the gap: we are already *better* than the frontier on Toffolis
(−8.4 %) and on the repetition factor, and still 3.9× behind purely on width.

**The likely shape of it.** Our decoder is `n = 70, degree = 12`, and `70 × 12 = 840` — the
exact number the forum gives for the syndrome bank it says cannot be shrunk. The frontier's
877-qubit build is 840 syndrome + ~37 co-live. So our 1,789 is plausibly the *same* syndrome
bank with **~949 co-live workspace** against their ~37: a decoder-fusion difference, not a
code difference. `dqi_src/peak_report.py` already exists to attribute the peak by phase and
is the next thing to run (it rebuilds the circuit, ~250 s).

**Leverage, one lever at a time (composite from 6.7267):**

| lever | ratio | composite |
|---|---:|---:|
| invsqrt 90 → 63 qubits | 13.157 → 18.80× | 7.58 |
| dqi 1789 → 877 qubits | 14.226 → **59.20×** | **10.82** |
| both | — | 12.19 |
| both + chemistry at 5.54× | — | 18.34 |

dqi is the largest single lever in the project and, unlike chemistry, it needs no training
pipeline: our Toffoli count and repetition factor already beat the frontier, so a 877-qubit
build of *our own* decoder would score 59.20× — above the public best.

**Measured (2026-10-28, `dqi_src/peak_report.py`, `compact_from=70`, 376 phases).** The
overall peak is **1,789 at `bm/r66`–`bm/r69`**, and its owners are:

| owner | wires | |
|---|---:|---|
| `syn` | **840** | syndrome bank — exactly the forum's "cannot shrink" number |
| `bb` | **444** | BM auxiliary (the B polynomial) |
| `sig` | 420–432 | BM working register |
| `stk` | 19 | |
| `dw`, `pq`, `dinv` | 12 each | |
| rest (`w0`,`wreg`,`m`,`tree`,`dcount`,…) | ≤ 12 each | |

So the split is **840 syndrome + 949 co-live**, and `bb + sig = 864` is **91 % of the
co-live**. The earlier guess was right about the numbers and wrong about the cause: this is
not slack that a compactor can find, it is the live state **Berlekamp–Massey inherently
carries across its `2r` iterations**. A build whose total is 877 therefore cannot be a
narrower BM — it must be a decoder that does not hold σ and B simultaneously, i.e. the
continued-fraction / Schur architecture the forum names as where the frontier's gains came
from ("BM → continued fractions → Schur"; jday96314's Schur decoder came out of round six of
an agent-driven search).

Note the symbolic build counts `toffoli = 396,089` where the shipped archive verifies at
`445,229`; the two counters disagree and the verifier's number is the one that scores.

**Consequence.** dqi's 4.161× squared-width lever (composite 10.82) is real but is a
decoder re-architecture, not a width trim. `--compact-from` and the borrow/stack knobs act
on phases that are not at the peak, so they cannot move the 1,789.

**Premise corrected (2026-10-28): invsqrt 90 → 63 is not compression, it is a different design.**
`code/invsqrt90/build_invsqrt90.py` reports `matches_notebook_digest: true` and
`matches_notebook_resources: true`, and `selected_schedule90.py` names the source:
Kozykorpesh's signed-curvature lifetime schedule, V3 / run 361677445, from kozykappa's
public notebook that scores 6.68108. **Our invsqrt is a byte-identical replica of a public
90-qubit design** — which is also why our composite 6.7267 sits right on that notebook's
6.68108. The 20.06× / 63-qubit frontier is mrmeows' *separate* design (`qcc-1375-open-circuits`,
Apache-2.0, generators included), built by "a chain of storage and uncomputation rewrites,
each checked on all 3.1 million legal inputs, the polynomial and the rotation angles never
changed". Two different architectures, each already tuned, so there is no incremental
90 → 63 path inside ours; the 27 wires come from the design, not from slack.

What that means for the numbers quoted above: `90 → 63` holding our own Toffoli content
would give 1,753.80 × 63 = 110,489 → 15.09× (composite 7.3); the full 20.06× / composite
7.58 also needs their 1,643.59 Toffoli-equivalent. Either way the action is to adopt a
public circuit, not to rewrite ours — and the same is true of the fact that everything we
currently submit already comes from one public notebook.

## A: mrmeows' 63-qubit invsqrt, verified locally (2026-10-28)

Dataset `mrmeows/qcc-1375-open-circuits` ("QCC 21 open circuits and sources", **Apache 2.0**,
updated 2026-10-27T02:12). Fetched `circuits/invsqrt.kmb` (459,891 B) into its own directory
with `LICENSE`, `circuits/NOTICE.txt`, `circuits/SHA256SUMS`; `sha256sum` matches the
dataset's own `SHA256SUMS` (`0766a111a76f241aa57fbbed466436d743be13436943ddb0628c291ca07d67d6`).
Note the single-file endpoint needs the slash percent-encoded
(`/api/v1/datasets/download/<ref>/circuits%2Finvsqrt.kmb`); unencoded 404s.

Our own official verifier (`verify_invsqrt_ext.py`, 200,540,928 shots):

```
status PASS   score 103,546.168164559   num_qubits 63
toffoli_equivalent 1643.5899708660158   toffoli_count 1348.4991469771198   max 1416
rotation_count 41   num_ops 46,610   reaction_depth 792   worst_error/tol 0.99932
```

Ratio **20.0565×** (baseline / 103,546.168), matching the author's claim to five decimals.
So the decomposition above is confirmed end to end: 63 qubits × 1,643.59 TE, where our own
build is 90 × 1,753.80.

| | invsqrt | dqi | dfthcsos | composite |
|---|---:|---:|---:|---:|
| current (submitted 6.68900) | 13.1573 | 14.2262 | 1.6261 | 6.7267 |
| **A: mrmeows invsqrt only** | **20.0565** | 14.2262 | 1.6261 | **7.7416** |
| C: all three from that dataset | 20.0565 | 49.3679 | 8.6121 | **20.4301** |
| C using the v9 components | 20.0565 | 55.28 | 9.45 | 21.8819 |

**A is +15.1 % on the composite and still below bronze**: bronze 8.21494 needs product
554.4, and A reaches 464.0. The same dataset's other two files are the step change — its
`NOTICE.txt` documents `dqi.kmb` as Özer's v8 Bezout–Schur, 883 qubits, cost
495,048,250,576.63 (49.37×), and `dfthcsos/` as 19 ANGLE circuits, C=59 merge, 8.6121×, all
Apache-2.0 with lineage. Neither has been fetched or verified here yet.

## C: adopting the public frontier components (2026-10-28)

Source: `mrmeows/qcc-1375-open-circuits`, **Apache 2.0**, 200 members. Every adopted file
was checked against the dataset's own `circuits/SHA256SUMS` before verification.

| component | file | sha256 vs SHA256SUMS | our official verifier |
|---|---|---|---|
| invsqrt | `circuits/invsqrt.kmb` | matches | **PASS**, 63 q, TE 1,643.59, 1,348.50 Toffoli, score 103,546.168 → **20.0565×** |
| dfthcsos | `circuits/dfthcsos/` (19 ANGLE + `submission.npz`) | 20/20 match | **PASS**, 423 q, 19,337 Toffoli, brot 8, J=4, error_2 19.976 mHa, λ_eff 47.635, score 389,636,552.07 → **8.6121×** |
| dqi | `circuits/dqi.kmb` (762 MB) | matches | **PASS**, 883 q, 519,152.196 Toffoli, rf 1.223014, 8/102,400 errors, score 495,048,250,576.63 → **49.3679×** |

The chemistry build is tighter than the forum's summary: 423 qubits / 19,337 Toffolis against
the reported 431 / 17,330. Its `E_gap = 3.914914` against the floor 3.914898 — **1.6e-5 of
margin**, the tightest of anything we have shipped, so it is worth re-running before any
final submission in case the scorer's float path differs.

Composite as adopted so far (invsqrt + chemistry, dqi still ours):

| | invsqrt | dqi | dfthcsos | composite |
|---|---:|---:|---:|---:|
| submitted | 13.1573 | 14.2262 | 1.6261 | 6.7267 |
| **invsqrt + chemistry adopted** | 20.0565 | 14.2262 | 8.6121 | **13.4943** |
| + dqi at 49.37× (pending) | 20.0565 | 49.3679 | 8.6121 | 20.4301 |

Two components alone take the composite from 6.7267 to **13.49**, past bronze 8.21494. This
is the public plateau, not original work: it should be shipped with the dataset's `NOTICE.txt`
and `LICENSE` preserved and attributed, which is what Apache-2.0 requires and what the
competition's reuse terms (thread 753316, still unanswered) are about.

### Also from the forum, not yet exploited

* **Eckart–Young floor `R·C ≥ 290` for 20 mHa** (307 for 15, 330 for 10). We ship
  `R·C = 1,950`, i.e. 6.7× the floor — the factorization is far larger than it needs to be,
  which is exactly why `outer`/`inner` state prep are 363→5,093 on the frontier shape;
* the 8-bit angle word has a **workspace cliff**: `temporary_qubits` roughly halves
  (543 → 304), which is most of the Q drop 661 → 431; plain rounding of a 10-bit fit to
  8 bits blows the gate (~111 mHa), so the refit must be quantisation-aware;
* a smaller dictionary at 10–11 bits with `R = 1` is a dead end (20.5–22.4 mHa best);
  `B = 255` stalls at 48 mHa; nine merged groups (R 10→9) reaches 22.1 mHa;
* `beta_1`/`S` remain closed by the verifier (`load_inputs`), and E_gap remains the
  binding gate — the frontier's E_gap is 3.9151 against a 3.915 floor, so it has *less*
  margin than ours (3.9265).

### Ruled out this session (2026-10-25, all measured)

| idea | result | why it fails |
|---|---|---|
| drop whole `rho` rows | 1 row: error_2 19.98 -> **32.98** | fit sits at 99.90% of a 20 mHa gate |
| drop whole `C` copy slices | 1 slice: 19.98 -> **345.88** | per-copy mass near-uniform (1.11/1.80/2.68) |
| exact Gram factorization of `T` | nuclear budget **4.07×** shipped | `lambda_eff` 1.96× worse even best-case |
| `Lambda` as an independent lever | `E_gap` is the binding gate | `E_gap` margin is 0.0116, `Lambda` has 6× |
| `beta_1` / `S` treatment | not controllable | `load_inputs` overwrites with baseline shifts |
| dqi `compact_from` knob | `Q` = 1789 at cf 0/20/40/70 | Toffoli strictly *worse* as cf falls |

None of these cleared the noise floor, so nothing was submitted. The v3 `W` refit
(+0.027%) remains built and verified but deliberately unsubmitted.

## Leaderboard result (submitted, scored)

`submission_v2.zip` was scored publicly as **6.68900** (ref 57598315, 2026-10-24) against
the local 6.7267 — a **0.56%** divergence, inside what the dqi phase-error resampling
predicts. That lands at **rank 181 / 668, top 27.1%**.

| cut | score | composite multiple | product-of-ratios needed |
|---|---:|---:|---:|
| bronze (rank 66) | 8.21494 | ×1.2281 | 304.4 → 563.8 |
| silver (rank 40) | 9.38045 | ×1.4024 | 304.4 → 839.4 |
| gold (rank 20) | 10.45081 | ×1.5624 | 304.4 → 1160.8 |

Note the bar moved: gold was 10.451 when this was written up. The three-axis ceiling with
every axis at its proven floor is **8.979**, so bronze is the only medal reachable without
a new factorization, and only if two axes improve together.

Two caveats that belong next to those numbers, because they are both thin:

* **invsqrt has almost no accuracy margin left.** `worst_error_over_tolerance` is
  **0.9996227435**, worst at `x = 32`. A 0.04% worse polynomial would fail. The 90-qubit
  schedule spends essentially the entire error budget; it is not a safe base to perturb.
* **dqi now has 3 phase errors per 102,400 shots** (the previous circuit had 0 value and
  0 phase errors). It still PASSes because `R(eps)` is computed from the measured rate
  and prices the leakage in — `R` is 1.2056 against the old 1.1913 — but the circuit is
  not clean, and on the leaderboard's different secret salt those 3 could be more.

### Refuted: buying anything back by cleaning up dqi's phase errors

The 3 phase errors look like a free 1.2% on dqi. They are not, because `R` is set by a
Clopper–Pearson bound at `log2_delta = -128`, and with `n = 102,400` even **zero**
observed errors leaves `eps_plus = 8.66e-4` and `R = 1.1913`. Measured directly off the
verifier's own `clopper_pearson_upper_bound` / `repetition_overhead`:

| errors | `eps_plus` | `R` |
|---:|---:|---:|
| 0 | 8.6606e-4 | 1.191260 |
| 3 (current) | 9.8391e-4 | 1.205584 |
| 7 | 1.1066e-3 | 1.219852 |

So going from 3 errors to 0 moves the dqi ratio 14.226 → 14.397, i.e. **×1.0120**, which is
**×1.0040 on the composite**. The `R > 1` is not a defect being paid for; it is the
statistical price of certifying an error rate from a finite sample, and it is the
verifier's choice of `n`, not ours.

Historic rows, kept for reference:

| task | cost | ratio | status |
|---|---:|---:|---|
| invsqrt (public) | 170,991.04 | 12.15× | replaced |
| dqi (public) | 2.298047e12 | 10.63× | replaced |

All three rows above come from a single end-to-end run of `four_addr/verify_archive.py`
against the final archive (no `--skip`), 2026-10-24 19:21:59, all PASS. The
`invsqrt.kmb` and `dqi.kmb` costs are identical to the earlier standalone runs, which
confirms those two members are byte-identical to the previously verified copies.

**Composite: 5.9443** (was 5.5527 before the four-address QROM port). Repacked as
`_scratch/submission.zip` (28 members, SHA-256 `6a278db1…`, 22.7 MB) with the new
chemistry family; `invsqrt.kmb` and `dqi.kmb` are byte-identical to the previously
verified copies.

**Status: below the top-10% bar, and not submitted.** Every lever that could reach the
bar has now been examined and closed (see the refuse sections below). The chemistry axis
is now provably exhausted: the four-address family is at its global optimum, measured,
and the remaining gap is 2.28x on invsqrt and dqi *each* — with both axes already near
their own structural ceilings.

The public checkpoint `rudrabhaskar/quantum-compilation-r6-verified-checkpoint`
(all 11 member SHA-256s matched its release receipt) scores 5.5418.

Top-10% line: **the leaderboard moved sharply on 2026-10-24.** Its 20 visible entries now
run from 17.714 down to 10.300, and every one of them sits above the best public notebook
(6.687). A 5.944 entry would not be in the visible table at all. The earlier "≈6.25 at
rank 25 of 256" estimate is stale and must not be used: at 6.25 you would now be several
times off the visible top-20 cut of **10.30**.

Re-read at 20:30 the same day: the 20 visible entries run 17.714 → **10.451**, so the bar
rose again. Reaching it from 5.9443 needs the composite to move 1.758x, i.e. the product of
the three ratios to go 210 → 1,141 (5.44x). At chemistry 1.626x that means
`invsqrt * dqi = 702`, against 129.2 now.

Because the score is a geometric mean, reaching 10.30 needs `invsqrt * dqi = 672` at the
best chemistry now available (1.626x). We have 129.2. Even fully porting the public
notebook's invsqrt and dqi work reaches only 6.67.

## The public 6.687 notebook, and what it actually buys

`kozykappa/quantum-ai-lb-score-6-68108` is the strongest public notebook available. Its
source is trustworthy: its `REFERENCE` constants
(invsqrt 2,076,774.2337 / dqi 24,439,474,177,519.6 / dfthcsos 3,355,599,588.3155) match the
baselines verified independently here, to 7+ digits. What it does **not** do is what its
title suggests:

* It reports chemistry at **661 qubits, 65,244 Toffolis, lambda_eff ≈ 47.53**. Running its
  own "chemistry loss is 12.716% lower" number backwards
  (`2.0499e9 = 47.532 * 65,244 * 661`) shows **lambda_eff is unchanged from 47.54**. The
  entire chemistry gain is the cheaper ANGLE-circuit synthesis (four-address QROM with a
  signed Givens gauge), i.e. T 72,378 -> 65,244 and Q 731 -> 661 on our factorization.
* Those figures are computed **inside their own notebook**. They have since been
  re-derived here by the official verifier (see "Ported and verified" below), which
  confirms them.

### Reproduced here: the optimiser does not move lambda_eff

`code/offset_opt.py` ports their `optimize_effective_norm` faithfully (L-BFGS-B on the
absorbed identity offsets, exact gradient through `eigh`, both E_gap and the LCU one-norm
in the objective, floors at 3.914898 and 49.0). On our factorization it gives:

    gap 3.922794 -> 3.915942,  Lambda 289.99 -> 289.70,  lambda_eff 47.537 -> 47.472  (1.0014x)

So the "pinned lambda_eff" conclusion from the earlier Adam refit **stands**; the correct
optimiser changes nothing either. Their per-stage description (`sparsify_columns` then
`optimize_effective_norm`, 3 stages, accept only on improvement) matches what was already
tried here in `chem_compress.py` + `chem_rotate.py`.

Where their real recurring gain lives is the **ANGLE circuit synthesis** (128-qubit
four-address QROM), worth T 72,378 -> 65,244 and Q 731 -> 661 on our numbers,
i.e. chemistry 1.325x -> 1.626x and composite 5.553 -> 5.944 (both now measured, below).
That is real, but it is ~0.39 on the composite, against a gap of ~4.4 to the leaderboard.

### Ported and verified here (2026-10-24)

#### invsqrt and dqi: built from source, byte-for-byte

`code/invsqrt90/` holds the eight modules the anazemcev source-exact notebook emits
(Apache-2.0), and `code/dqi_src/` holds its fourteen-file dqi source chain. Both were
rebuilt here rather than copied as binaries, and both reproduce the published artifacts
exactly:

| artifact | published | ours | evidence |
|---|---|---|---|
| invsqrt circuit SHA-256 | `97f12b10…` | `97f12b10…` | same bytes |
| invsqrt resources | 90 q / 1,449 T / 44 rot | 90 / 1,449 / 44 | asserts in `build_invsqrt90.py` |
| dqi resources | 1,789 q / 445,410 T | 1,789 q / **445,229** T | `build_report.json` |

The invsqrt build is four stages, and each intermediate digest matches the notebook's
record: raw schedule `5cb58379…`, after exact-domain compilation `343ae65e…`, final
`97f12b10…`. The only inputs are the `.kmb`-free Python sources and a 44-entry integer
calibration table, so this is reproducible without any Kaggle dataset.

Verified with the official verifiers:

| task | cost | ratio | Q | T | notes |
|---|---:|---:|---:|---:|---|
| invsqrt | **157,841.92** | **13.157×** | 90 | 1,449 | PASS, `worst_error_over_tolerance` **0.99962** |
| dqi | **1.717915e12** | **14.226×** | 1,789 | 445,229 | PASS, `R = 1.2056`, **3 phase errors** / 102,400 |

Both caveats are material. The invsqrt circuit spends **99.96% of its error budget**
(worst at `x = 32`), so it cannot be perturbed. The dqi circuit leaks phase on 3 of
102,400 shots where the old one leaked on none; `R` absorbs it statistically and it
still passes, but the margin is now a measured quantity rather than zero.

#### The chemistry family: built from scratch

`code/four_addr/` re-implements the construction from scratch (Apache-2.0 technique from
the same notebook) and applies it to our C=30 factorization. The signed Givens gauge is
exact (max component drift 7.8e-16, and `bit9 == bit8` holds in all 298,200 words and all
298,200 delta words). The official verifier scores it:

| build | Toffoli | qubits | chemistry | composite |
|---|---:|---:|---:|---:|
| `_scratch/gauge/own_J3c/` (clean-room, aligned allocator) | 65,244 | 661 | **1.6261x** | **5.9443** |
| earlier clean-room, before the allocator fix | 67,544 | 661 | 1.5707x | 5.8760 |

**Composite 5.5527 -> 5.9443**, and the clean-room build now matches the notebook's own
65,244 / 661 / 1.626x exactly (same chemistry score to the last digit), so those figures
are confirmed by the verifier rather than merely self-reported.

The 2,300-Toffoli gap in the first attempt was fully diagnosed and is now closed. It was
46 CCX per circuit = 23 pebble erasures x 2 (erase + recompute), and the cause was one
line in the depth function: the notebook's `need()` only recurses into a branch that
still has children (`if children(c, j)`), so leaves — which use `direct_leaf_ands` and
need no carry wire — do not count toward the required depth. My first version counted
every child, which made `need` one too large everywhere and made the walk pebble where
it did not have to. With that filter fixed the walk never erases, and 5 scratch wires
suffice — exactly as in the notebook.

### Why this idea cannot be pushed further (measured, not projected)

The family is parameterised by (s, J) under the 128-qubit cap

    qubits = sel + J*brot + 9*J*(2**s - 1) + scratch,   sel=12, brot=10, scratch=5

and the cost is `T_angles ~ 2*K*(H + 9*J*(2**s-1) + cleanup)`, `H = A/2**s`,
`K = ceil(75/J)`. Exhaustive sweep of every feasible (s, J) — including the
non-banked s=0 branch, which nobody has published:

| s | J | K | qubits | per-circuit | T_angles | chemistry |
|---:|---:|---:|---:|---:|---:|---:|
| **2** | **3** | 25 | 128 | 1,127 | **56,350** | **1.629x** |
| 0 | 11 | 7 | 127 | 4,090 | 57,260 | 1.606x |
| 1 | 5 | 15 | 112 | 2,116 | 63,480 | 1.468x |
| 0 | 10 | 8 | 117 | 4,090 | 65,440 | 1.429x |

**s=2, J=3 is the global optimum of the family, and it is the point the notebook
already uses.** It sits 2.3% from the continuum optimum of
`T ~ 150A/(J*u) + 1350u` subject to `J*(9u+1) <= 111`, `u = 2**s`. Two independent
checks say there is nothing left here:

* **The bank is incompressible.** `banked_lab.difference_tables` compresses the
  difference bank by GF(2) column rank; on this data the rank is *exactly* the raw
  width at every (s, J) tried (27/27, 81/81, 135/135, 63/63, 189/189, 315/315), with
  zero zero-columns. Neither contiguous nor similarity-greedy row regrouping helps.
  So no wires can be bought back to raise s or J.
* **The 128-qubit cap binds at the optimum.** s=3 needs 9*J*7 bank wires; even J=1
  costs 90 qubits and J=2 costs 163. s=2, J=4 costs 165.

Two things the notebook does not exploit, both worth checking and both closed:
dropping the word registers to 9 wires (bit9 is redundant with bit8 — true, but it
only moves 128 -> 125, still not enough for s=2 J=4), and carrying the bank across
circuits (not allowed: every ANGLE circuit must return its workspace to |0>).

### What that leaves

At chemistry 1.6261x, reaching the visible top-20 cut of 10.30 needs
`invsqrt * dqi = 672`; we have `12.1455 * 10.6349 = 129.2`. That is **2.28x on
invsqrt and dqi each**.

The public notebook's invsqrt fusion (12.15 -> 16.63) and dqi (10.63 -> 10.98) recover
only 1.37x and 1.03x, giving composite 6.67 — still 3.6 short.

**Both axes are also near their structural ceilings, which closes the composite too:**

* dqi cost is `R * Q^2 * T` with `R = 1.1913` at `log2_delta = -128`. Even `R = 1.0`
  exactly — zero statistical safety margin, which would fail verification — gives only
  12.67x at the current Q=1818, T=583666.
* invsqrt has `worst_error_over_tolerance = 0.910`, i.e. 1.099x of accuracy headroom.
  Spending all of it gives at most ~13.35x at Q=95.

Their product at those absolute ceilings is **169.1, against 672 needed — a factor 3.97
short.** So no combination of the remaining chemistry, invsqrt and dqi levers reaches the
visible top-20, and it is not close. Closing the gap needs a different circuit family on
invsqrt and dqi (fewer qubits, not fewer Toffolis: `Q^2` dominates dqi, and Q=95 with
C=1800 dominates invsqrt), which is a research problem, not a tuning problem.


## The `W` / `H2_I` freedom: measured, and closed by a numerical bound (2026-10-24 evening)

This is the one structural lever the earlier passes never touched, and it is worth
recording because it *looks* like a large opening and is not.

The verifier freezes `beta_1` and `S` (`load_inputs` replaces both with the baseline's
`fixed_shifts`) but it does **not** freeze `H2_I`. `native_inputs` computes

    W_rc = H2_I_rc - sum_b w_b

so all 1,950 entries of `W_rc` are a *submitted* array. `W` enters three separate places:

    lambda_h2  = 0.25 * sum_rc (|W_rc + s_rc| + L1_rc)^2,    s = sum_b w
    h1_prime   = C0 - sum_rc W_rc X_rc,      C0 = h1_thc - (eta/2)S + beta_1 I
    E_SOS      = ecore - eta*beta_1 - (lambda_h1 - sum lambda) - 0.5*||W||^2

and `error_1` vanishes identically when the submitted one-body rows are the exact
eigenvectors of `h1_prime` (`reconstruct_h1_thc` inverts the same identity, so the
`sum W X` terms cancel). So `W` is a genuinely free 1,950-parameter knob on `Lambda`
with no accuracy cost. What it cannot do is move `Lambda` down:

* **`h1_prime` can be nulled exactly.** A linear least-squares solve over the 2,926
  symmetric entries of `C0` against the 1,950 `X_rc` reaches a residual of 0.163, i.e.
  1.8e-3 relative, and takes `lambda_h1` from 13.17 to **1.19**. That is real, and it
  makes `error_1 = 0` exactly.
* **But the E_gap floor is the binding constraint, and it is one inequality.** With
  `A = ecore - eta*beta_1 = -21581.141` and `E_upper = -22140.27335`,

      E_gap > 3.914898  <=>  (lambda_h1 - sum lambda) + 0.5*||W||^2 > 563.0474

  At the submitted point that quantity is **563.059** — over by **0.0117**, a margin of
  0.002%. `E_gap` is not "near" its floor by accident; the solution is pinned on it.
  Of that 563.06, `0.5*||W||^2 = 539.82` and `lambda_h1 - sum lambda = 23.24`, so 96%
  of the budget is the weight norm.
* **`Lambda` therefore has a numerical floor.** Since `lambda_h1 >= sum|negative eig|`
  and `lambda_h1 - sum lambda = 2*sum|negative eig|`, the constraint gives
  `lambda_h1 >= 0.5*(563.0474 - 0.5*||W||^2)`, so

      Lambda >= 0.5*max(0, 563.0474 - 0.5*||W||^2) + 0.25*sum_rc (|H2_I| + L1_rc)^2

  Minimising that proxy over `W` (60 L-BFGS-B restarts) gives **291.93**, against the
  current 293.53. With `E_gap` pinned at its floor, `lambda_eff ~ 2.7980*sqrt(Lambda)`,
  so the reachable floor is **lambda_eff 47.81 against the current 47.850 — 1.001x.**
  The chemistry axis is closed by arithmetic, not by lack of ideas.
* **The naive direction fails hard.** Scaling the submitted `W` up by 2% to buy `E_gap`
  room takes `lambda_eff` from 47.85 to 81.09; by 10% to 258.5. Setting `H2_I = 0`
  (the `lambda_h2` floor, `0.25*sum L1^2 = 139.25`) gives `lambda_h1 = 236.3` and
  `lambda_eff = 317.2`. The joint optimum is the submitted point.

### The `W` freedom, measured with every gate enforced (2026-10-25)

The 291.93 above came from a *proxy* bound. `code/chem_W_opt.py` instead optimises the
real objective — `lambda_eff` from the verifier's own formulas — over all 1,950 `W`
entries with autograd through `torch.linalg.eigvalsh`, and enforces **all** the gates:
`E_gap > 3.914898` with a strict margin, `Lambda > 49`, **and `error_1 <= 20 mHa`**. It
refuses to run unless its model reproduces `evaluate_words` exactly (it does, to 1e-9 on
all four of `lambda_eff`, `Lambda`, `E_gap`, `error_1`).

`error_1` is the gate that actually binds `W`, and it had been left out of every earlier
attempt. `h1_prime` depends on `W`, so moving `W` moves the operator the *fixed* one-body
rows encode: the first optimiser that ignored `error_1` "found" `lambda_eff` 47.7587 but
paid for it with `error_1` 29.89 mHa, over the gate. With `error_1` enforced:

| `W` | `lambda_eff` | `Lambda` | `E_gap` | `error_1` | passes |
|---|---:|---:|---:|---:|:--:|
| submitted | 47.85037 | 293.5256 | 3.926532 | 4.15 mHa | yes |
| optimised | **47.81022** | 293.5288 | 3.919858 | 9.48 mHa | yes |

**Gain ×1.00084, composite 6.689 → 6.6909.** The proxy 1.001× and the measured 1.0008×
agree, from opposite directions. The chemistry axis is closed: there is no `W`.

### What this refutes about the small-B story

`0.25*sum L1(w)^2` is the `lambda_h2` floor at fixed `w`, and it is **139.25 at B=60 but
68.93 at B=30** — so a smaller `B` does halve the best term in `Lambda`. The earlier
"small B is dead" verdict was right for a different reason than recorded: the `B=30`
checkpoint's problem is `h1_prime` going negative-semidefinite (`lambda_h1 = 410`), not
its weights. That looked like exactly what the `W` freedom could repair. It cannot:

| factorization | `lambda_h2` floor | `lambda_h1` as trained | `lambda_h1` after `W`-nulling | best `Lambda` | `lambda_eff` |
|---|---:|---:|---:|---:|---:|
| B=60 (submitted) | 139.25 | 13.17 | 1.19 | **293.5** | **47.85** |
| B=30 checkpoint | **68.93** | 410.08 | 249.50 (W-optimised) | 507.8 | 503.9 |

The `B=30` `X` bank spans too little of the symmetric space: the nulling solve needs
`W` of order 1e13, i.e. it does not exist. Constrained optimisation in `W` settles at
`Lambda = 507.8`, ten times worse. So `B` cannot shrink through this route either.

**Correction (2026-10-25).** The two small-`B` entries above blame different mechanisms
and the "spans too little of the symmetric space / needs `W` of order 1e13" story is the
weaker of the two. Re-measured on the *submitted* point, the binding quantity is the
`E_gap` inequality and it is satisfied almost entirely by `||W||^2`, not by the `X` span:

| quantity | value |
|---|---:|
| `p = sum|negative eigenvalues of h1_prime|` | 11.621 |
| required `0.5*||W||^2 > 563.0474 - 2p` | 539.805 |
| actual `0.5*||W||^2` | 539.817 (margin 0.0116) |
| `||W||^2` if `W` cancelled `sum_b w_b` (`H2_I = 0`) | 521.990 → `0.5*||W||^2` = 261.0 |
| `sum_b ||w_b||_1` per `(r,c)`, i.e. `L1` | mean 0.401, max 3.887 |

So **`H2_I != 0` is forced by `E_gap`**: cancelling `sum_b w_b` exactly leaves the gap
278.8 short. That is why `lambda_h2` sits at 280.4 against its `H2_I = 0` floor of 139.2 —
the solution is paying for the gap in `H2_I`, deliberately and correctly. The earlier
"`W` of order 1e13" phrasing described a solve that was never the operative constraint.

### The `lambda_h2` budget, itemised — and why spreading the weights does not help

`lambda_h2 = 0.25 * sum_rc (L1_rc + |H2_I_rc|)^2` splits exactly:

| term | value | share of `lambda_h2` |
|---|---:|---:|
| `0.25 * sum_rc L1_rc^2` | 139.2473 | 49.7% |
| `0.25 * sum_rc H2_I_rc^2` | 26.3033 | 9.4% |
| `0.5 * sum_rc L1_rc |H2_I_rc|` (cross) | 114.8059 | 40.9% |
| **total** | **280.3565** | |

The tempting observation is that `sum L1^2 = 556.99` while an even spread of the same
`sum L1 = 781.75` over 1,950 entries would give only `313.40` — a 1.78× penalty, so
"spread the weights" looks worth ~40 in `Lambda`. **It is not, and the reason is that
`sum L1` cannot come down.** Per `(r,c)`:

* every one of the 1,950 `(r,c)` pairs has **57–60 nonzero `w_b`** (mean 59.3), because
  `rank(X_rc) = 57–60` — a rank-`k` matrix needs at least `k` atoms, so the atom count is
  already minimal, and lowering `B` would truncate the rank, not the weights;
* `L1_rc` sits at **1.0438× its own nuclear-norm floor on average** (min 1.0000, i.e.
  exactly on it; max 2.06). `sum L1 = 781.75` against `sum nuclear = 758.58` is a **3.1%**
  excess. So the weights carry essentially all of the mass they must.

The 1.78× concentration is variation *between* `(r,c)` pairs, not redundancy *within* one,
and redistributing it would have to spread the E_gap-carrying `H2_I` mass too. Even
driving every `L1` to its nuclear floor exactly, with `H2_I` frozen, only takes
`lambda_h2` 280.36 → 274.97 (and that ignores the resulting `h1_prime`/`E_gap` change).
Every route into the 139.25 floor runs through `H2_I → 0`, which `E_gap` forbids.

**Conclusion: the chemistry axis is closed at every level I can measure** — `W` ×1.0008,
`brot` (no headroom, `error_2` already at 99.9% of its gate), `L1` (3.1% from its floor),
`B`/`R`/`C` (rank and `tr(M)` arguments, now confirmed by direct row/slice drops), `Lambda`
(the `E_gap` inequality), and the exact Gram factorization of `T` (4.07× the nuclear
budget). Measured rather than projected in every case; see the sections below.

### A floor on `lambda_h2` that no factorization can beat

`code/chem_lower_bound.py`. Three facts combine:

1. `lambda_h2 = 0.25 * sum_rc (L1_rc + |H2_I_rc|)^2` with both terms nonnegative, so
   `lambda_h2 >= 0.25 * sum_rc L1_rc^2` regardless of `W`.
2. `X_rc = sum_b w_b P_b` with `P_b = u_b u_b^T` a **rank-one projector**, so
   `||P_b||_F = 1` and `||X_rc||_F <= sum_b |w_b| = L1_rc`. Hence
   `lambda_h2 >= 0.25 * sum_rc ||X_rc||_F^2`.
3. The verifier's `error_2` target is `M = tei + 0.5(S x I) + 0.5(I x S)`, and
   `sum_rc ||X_rc||_F^2 = tr(sum_rc X_rc x X_rc)`. So `sum_rc ||X_rc||_F^2` is pinned to
   `tr(M)`, which depends only on the FCIDUMP and the verifier's fixed shifts.

Measured: `tr(M) = 53.297001` and the submitted `sum_rc ||X_rc||_F^2 = 53.323865` — they
agree to 0.027, i.e. to `error_2`'s own precision. Therefore

    lambda_h2 >= 0.25 * tr(M) = 13.324250,   Lambda >= 13.32 + sum|lambda_j|,

**independent of `R`, `B`, `C`, `brot`, or any future factorization.** The submitted
`lambda_h2` is 280.36, so there is nominally 21× of `Lambda` headroom. It is not reachable:
the `E_gap` term forces `0.5*||W||^2 ~ 563`, and `W` re-enters `lambda_h2` through
`|H2_I| = |W + sum_b w_b|`. A tighter variant uses the nuclear norm
(`||P_b||_* = 1` gives `L1_rc >= ||X_rc||_*`), which on the *current* directions gives
`0.25 * sum_rc ||X_rc||_*^2 = 134.70`. Both are floors; neither is attainable while the
gap is satisfied.

### The exact spectral (Gram) factorization of `T`: refuted, with the arithmetic

`T` pairs index pairs, so `T_{pq,st} = tei_pqst + 0.5 S_pq d_st + 0.5 d_pq S_st` is a plain
`(N^2, N^2)` matrix, and `sum_rc X_rc (x) X_rc = sum_rc v_rc v_rc^T` with `v_rc = vec(X_rc)`
is exactly a Gram matrix. That gives an **exact, fit-free** factorization:
`T = sum_i c_i phi_i phi_i^T` with `X_i = sqrt(c_i) * unvec(phi_i)`, `error_2 = 0`, `C = 1`,
and `K = rank(T)` terms. Unlike the old "direct SVD" note at line ~689 (which truncated to
rank-1 atoms and was rightly abandoned), this form *is* realizable by DF-THC — each `X_i` is
a genuine matrix. So it was worth measuring. Measured, it fails, and not marginally:

| `K` | `error_2` | `0.25 sum nuc^2` | vs shipped 139.2473 |
|---:|---:|---:|---:|
| 65 | 547.35 mHa | 511.3310 | 3.672x |
| 200 | 77.17 mHa | 557.3651 | 4.003x |
| **300** | **16.70 mHa** | **564.7549** | **4.056x** |
| 400 | 4.13 mHa | 566.4045 | 4.068x |
| 1950 | 0.60 mHa | 567.0920 | 4.073x |

Accuracy is fine — `K = 300` already passes with 17 mHa, and `K = 400` passes with 5x margin.
The wall is `lambda_h2`. Writing `u_b` = eigenvectors of `X_i`, `w_b` = eigenvalues gives
`sum_b w_b u_b u_b^T = X_i` with `L1_i = ||X_i||_*` and `sum_b w_b = tr X_i`, so
`lambda_h2 = 0.25 sum_i (||X_i||_* + |W_i + tr X_i|)^2 >= 0.25 sum_i ||X_i||_*^2 = 564.75`
at `K = 300` — **4.06x the shipped nuclear budget**, because eigen-matrices of a two-electron
integral are dense and full-rank (`sum|tr X| = 30.9` is negligible). Best case for `E_gap`
(void `W = 0`, all of the forced `563.047371` supplied by `-p`): `p = 281.52`,
`Lambda = 1127.80`, `lambda_eff = 93.889` — **x1.962 worse**, composite x1.252 worse, before
counting ANGLE Toffoli, which rises 5.75x (`N + R*B = 22,876` rows vs 3,976). Dead.

### `R`, `B` and `C` enter `T` through two *different* products (measured)

`code/rbc_marginal.py`. Givens is `2(N-1)(2 beta - 1)` — independent of all three. Only the
state-preparation terms move, and they key off different products:

* ANGLE table size = `N + R*B` (not `R*C`): `U` is `(R,B,N)` and `angle_words` is `(3976,75)`.
* Heuristic `T` and `Q` key off `R*C` and `B`.

Marginal value at `brot = 10`, from shipped `(65,60,30)` `= T_h 8,794 / Q_h 631`:

| change | heuristic `T` | `Q` |
|---|---:|---:|
| drop 1 row (R=64) | -0.830% | -0.158% |
| drop 9 rows (R=56) | -7.403% | -0.158% |
| B 60 -> 48 | -1.092% | 0.000% |
| B 60 -> 32 | -2.922% | -5.071% |
| C 30 -> 16 | -24.915% | 0.000% |
| C 30 -> 8 | -39.402% | -0.475% |

`C` is the strongest knob and the only one with real leverage. But it is closed by measurement:
dropping **one whole copy slice** (`code/drop_copy_probe.py`) takes `error_2` 19.98 -> **345.88
mHa**, and dropping one whole **row** (`code/drop_rho_row_probe.py`) takes it to **32.98 mHa**.
Per-copy mass is nearly uniform (min 1.1127, median 1.7978, max 2.6828), so there is no
slack copy to remove. The fit sits at 19.9799 of a 20.000 mHa gate — **99.90% of budget** —
so no structural removal of any kind can pass.

**Correction to `drop_rho_probe.py`:** it zeroed individual `(rho,c)` entries and reported
`R = 65 - drop//30`, which *fabricates* a shrinking `R`. Zeroing an entry changes nothing about
the circuit — `R`, `C`, `T` and `Q` are all unchanged. That probe measures `error_2` only.
The row and copy probes above are the cost-bearing versions; the fix is applied in the file.

### `Lambda` is not an independent lever: the gate that binds is `E_gap`

`code/lambda_egap_identity.py`, measured on the shipped archive:

    Lambda = 293.525553   (floor 49 -- 6x headroom, NOT binding)
      lambda_h2 = 280.356471 (95.51%)
      lambda_h1 =  13.169082
    E_gap  = 3.926532     (floor 3.914898 -- 0.0116 of margin, BINDING)
    0.5*||W||_F^2 = 539.816863      0.25*||W||_F^2 = 269.908432
    ||H2_I - W||_F^2 = 521.989501   ||H2_I||/||W|| = 0.3122

So the note "`Lambda > 49`, 6x headroom" is true and irrelevant: `lambda_eff` is
`sqrt(E_gap(2 Lambda - E_gap))`, and with `E_gap` pinned at a floor of 3.914898 the only way
to lower `lambda_eff` is to lower `Lambda`. Pushing `E_gap` to its floor with `Lambda` fixed
is worth only x1.001475. The two routes (`E_gap` up, `Lambda` down) trade off one-for-one
through `0.5||W||^2`, which is why `H2_I -> 0` cannot help: zeroing `H2_I` drops
`lambda_h2` toward `0.25||L1||^2 = 139.25` but costs `E_gap` the full `539.82` of
`0.5||W||^2`, and re-supplying that `W` puts it straight back into `lambda_h2`.

### Why "99.99% of the mass kept" still fails: `error_2` is first-order in `dX`

`code/rank_per_rc.py` measures the per-entry rank distribution: exact ranks are 56-60
(median 59), and truncating every `X_rc` to `k = 50` keeps **99.9934%** of the Frobenius
mass (99.9991% at `k = 56`). Meanwhile `rank_probe.py` measures `error_2 = 90.99 mHa` at
`k = 48` and 23.10 mHa at `k = 56` — both failures — while the energy kept is 99.99%.

That is not a contradiction, it is the structure of the objective:

    sum X'(x)X' - T = sum (X+dX)(x)(X+dX) - sum X(x)X  =  sum (dX(x)X + X(x)dX) + sum dX(x)dX .

The **cross term is first order in `dX`**, so error_2 ~ 2||X|| ||dX|| while the discarded
energy is ~||dX||^2. A relative tail of 6.6e-5 in energy is ~8e-3 in amplitude, and
2 * 0.008 * ||X|| is already the whole gate. So **"energy kept" is the wrong progress
metric for this problem** — any pruning argument must be made on the cross term. This is
why every structural truncation (rows, copies, ranks) fails while looking harmless:
the fit sits at 99.90% of a 20 mHa gate, so the *first-order* perturbation has no room in
which to be small. The exact-Gram route avoids this by construction (error_2 = 0) and dies
on `lambda_h2` instead.

### Closed by the verifier, not by chemistry: `beta_1` and `S`

`verify.load_inputs` calls `params.update(load_fixed_shifts())` **before** deriving `S` and
`beta_1` (`data/verifiers/dfthcsos/verify.py:218`). The submitted `H1_I` and `Hsym_lower`
are discarded; the baseline's are used instead. So despite `beta_1` appearing in
`E_SOS = ecore - eta*beta_1 - (...)`, and despite `eta = 113` multiplying it, it is **not a
free parameter**. The same holds for `S` and therefore for the whole `h1_prime` shift.
Do not spend effort on the `beta_1` treatment — the verifier does not read it.

### Where the chemistry `T` and `Q` actually are

The submitted archive holds **25** ANGLE circuits (`angle_k_0` .. `angle_k_24`, confirmed by
listing `_scratch/submission.zip`) at `(R,B,C,brot) = (65,60,30,10)`, `J = 3`. An earlier
note in this file said "8", which came from the stale `_scratch/final_dfthc.json` (8 circuits,
2.53e9) rather than from the shipped archive (25 circuits, 2.063608e9). The 25 count is the
consistent one: `ceil(75/3) = 25`.
`T = 58,996` (ANGLE 50,202 + heuristic 8,794), `Q = 731` (persistent 188 + temporary 543).

* **`Q`**: the ANGLE circuits use only **123** qubits, and `Q` takes
  `max(heuristic_temporary, angles_temporary)`, so the ANGLE part is free in `Q`. All 543
  temporary qubits come from `inner_state_prep`, of which **539** is
  `temporary_for_qroam = ceil(log2(RC*ceil((B+1)/2^k2))) + (2^k2 - 1)*b2` with
  `b2 = ceil(log2 B) + chi + 1 = 17`. That `17` multiplier is the whole `Q` lever.
* **`T`**: each ANGLE circuit is 3,974 Toffoli — 2,700 of it the 75 Givens rotations over
  4 passes (`4*(N-1)*9`), 1,274 the QROM angle load/uncompute. 8 × 3,974 × 2 passes =
  63,584 structural, 50,202 after conditional-gate accounting. The heuristic `T` is only
  8,794, so **85% of chemistry `T` is the angle circuits**.

Both of these scale with `R*B` (the angle table) and with `B` (the `b2` in `Q`), which is
why a smaller factorization is the only route to a large chemistry gain — and why it has
to pass `error_1`, `error_2`, `E_gap` and `Lambda > 49` simultaneously, which is what every
small-`B` attempt so far has failed to do.

### The brot <= 7 cliff: the model pays, the accuracy gates do not

`scoreinfra/costs_prepare.py` has a real cliff: `find_qroam_size_to_minimize_global_temp_ancilla`
requires `temp < beta*(N-1)`, so at `brot <= 7` the inner QROAM block shrinks and the
temporary qubits fall from **543 to 272** at `(65,60,30)`, at a cost of `T_heur`
8,794 → 11,522. Combined with a smaller `B` the *model* would allow up to 9.8x
(`chem_ceiling.py`: `B=12, brot=5` → `T=21,200`, `Q=337`), and even `B=32, brot=6`
gives 4.84x.

Measured with the verifier's own `evaluate_words` on the shipped words
(`code/brot7_probe.py`), requantising 10 → 7 bits fails the gates by two orders of
magnitude:

| brot | error_1 (mHa) | error_2 (mHa) | lambda_eff | passes |
|---:|---:|---:|---:|:--:|
| 10 | 4.15 | 19.98 | 47.850 | **yes** |
| 9 | 938.37 | 186.75 | 59.182 | no |
| 8 | 1118.99 | 229.19 | 61.903 | no |
| 7 | 1565.93 | 357.20 | 68.189 | no |

This confirms the earlier "brot is a cliff, not a knob" note and extends it: the words
are co-fitted to their exact 10-bit values, so reaching `brot <= 7` needs a
quantization-aware refit of the whole factorization, not a rounding. Given that the
`Lambda` bound above holds for *any* `w` (it is a property of the constraint, not of
this factorization) and that `lambda_h2` at `brot`-independent `w` cannot go below
139.25 without also fixing `h1_prime`, the 2.96x the model shows at `brot=7` is not
reachable. **Closed.**

**The `error_2` budget is already spent at `brot = 10`** (2026-10-25). It is worth being
precise about *what* the requantisation table measures, because the obvious reading is
wrong. `submission.npz` stores `U` already decoded from its own 10-bit words
(`max|U - decode(angle_words, brot=10)| = 6.4e-16`), so comparing "stored `U`" against
"decoded words" measures nothing — they are the same object. The real decomposition is
that `error_2 = 19.9799 mHa` at `brot = 10` is **99.90% of the 20 mHa gate before any
quantization loss**, and each bit given up multiplies it:

| brot | error_1 (mHa) | error_2 (mHa) | fraction of 20 mHa gate |
|---:|---:|---:|---:|
| 10 | 4.15 | 19.9799 | 99.90% |
| 9 | 938.37 | 186.75 | 934% |
| 8 | 1118.99 | 229.19 | 1146% |
| 7 | 1565.93 | 357.20 | 1786% |

So the refit this section used to call for cannot help either: there is no quantization
headroom to trade for `brot` at all, because the fit sits on the gate already. Lowering
`brot` needs `error_2` to *fall* first, and `error_2` is the two-body residual of the
factorization itself — bounded below by the `tr(M)` argument in the section above.

### invsqrt and dqi: what the public field has that we do not

Three invsqrt builds exist in public material, and they rank cleanly. Cost is
`Q * (Toffoli + T/4)` with the 44 rotations priced by the verifier's own
`rotation_t_count`:

| build | qubits | Toffolis | rotations | cost | ratio |
|---|---:|---:|---:|---:|---:|
| ours (`invsqrt.kmb`, verified) | 95 | 1,472 | 44 | 170,991 | 12.146 |
| `alexchilton` v6 | 93 | 1,470 | 44 | ~165,026 | 12.58 |
| `anazemcev` source-exact V4 (**best**) | **90** | **1,449** | 44 | **~156,900** | **13.24** |

The 90-qubit build is a kozykappa checkpoint schedule plus anazemcev's full-domain
exact compiler plus a certified fixed angle calibration (`stage_ref`'s `F2` 10 → 9,
borrowed `x[21]`, cursor-sharing schedules). Going from our 95 to 90 qubits is
**1.069x** on this task, the single largest invsqrt move available.

For dqi:

* **The port** (Gu/Kemeny/Yelin/Lukin, arXiv:2610.22367, Apache-2.0): truncated
  Berlekamp–Massey with contracted Forney, **1,789 qubits / 445,410 Toffolis**, cost
  **1.72e12** against our 2.298e12, i.e. **1.336x**. Our dqi is a 1,818-qubit circuit
  at `T = 583,666`; the port's win is mostly `T`, since `Q^2` barely moves.
* The tower-42 arithmetic route (`avazaku`, and `lightningv08`'s Memory & Algebra) is
  *strictly worse* for the competition: it takes mean Toffoli equivalents from 5.725M
  to 4.503M (1.27x), while the Berlekamp–Massey circuit already sits at **583,666**.
  Rank-42 is a better multiplier, not a better decoder. **Closed.**

Best combination now actually available: invsqrt 90q + dqi port + our chemistry
= **13.157 × 14.226 × 1.626 → composite 6.7267.** Both ports are now built and
verified (see "Ported and verified" above). The ratio product is **304.4** against the
**1,141.4** the 10.45081 bar needs, so **3.75x** is still missing.

The imports are recorded below because they were reproduced byte-for-byte, which is
worth more than their headline numbers:

| stage | notebook's recorded SHA-256 | ours |
|---|---|---|
| raw 90-qubit schedule | `5cb58379…` | match |
| after exact-domain compilation | `343ae65e…` | match |
| final calibrated circuit | `97f12b10…` | match |

Resources came out at **90 qubits / 1,449 Toffolis / 44 rotations**, exactly as
published, and `code/invsqrt90/build_report.json` records the evidence. The dqi
source chain gave **1,789 qubits / 445,229 Toffolis** against the paper's 445,410
(ours is 181 better) at `R = 1.2056`.

The arithmetic of the gap, precisely. The ratio product is 210.0 now and the bar needs
`10.45081^3 = 1141.4`, a factor of **5.434** on the product. Both ports take it to
290.7, so they recover 1.384 of that 5.434 and leave **3.93x**. Closing it on chemistry
alone would need a chemistry ratio of **6.38** — a chemistry cost of 0.526e9 against the
2.064e9 now, i.e. 3.9x better than the four-address family, which the `Lambda` bound
above shows is impossible (that bound caps chemistry at 1.627x). So the ports are worth
doing for their own sake, not as a route to the bar.

### The ceiling with every axis at its proven floor

Putting all three tasks simultaneously at the best bound this analysis can prove gives
composite **8.98**, still under the 10.451 bar:

| task | floor used | cost | ratio |
|---|---|---:|---:|
| invsqrt | v6's 93 qubits/1,470 T, then all of the 1.099x accuracy headroom spent | 150,160 | 13.83 |
| dqi | `R = 1.0` (zero safety margin) at the port's 1,789 qubits | 1.868e12 | 13.08 |
| dfthcsos | `Lambda = 49.0`, `E_gap = 3.9149` exactly ⇒ `lambda_eff = 19.192` at the current `T*Q = 43.7e6` | 8.386e8 | 4.00 |

**(13.83 × 13.08 × 4.00)^(1/3) = 8.979.** Two of those three floors are themselves
infeasible (dqi at `R = 1.0` has no statistical margin and would fail verification;
`Lambda = 49.0` is rejected because the floors are strict). So 8.98 is a generous
*over*-estimate of what tuning reaches.

The visible top-20 must therefore be sitting on something structurally different, not on
better tuning: the best *observed* public score anywhere is 6.68108 lower down the table
(they overtake us only via the medal cutoff), the whole public field ranges 6.28–6.68,
and the bar is 10.45. Getting from 8.98 (a bound) to 10.45 needs
a further 1.16x on the product, which no combination of the levers documented here
supplies. Reaching the visible table requires a new circuit family on at least one task
that is not a better gate or a better schedule — most plausibly, a chemistry
factorisation that genuinely reaches `Lambda` near 49 while keeping the accuracy gates,
since that single axis has 2.49x of provable but unrealised headroom.

**Update after the score came back (2026-10-25).** The scored 6.6890 vs the bound 8.979
is a 1.34× gap, and every section below closes a different candidate for it. What the
2026-10-25 work adds is that the chemistry gap is *not* a matter of finding better `W`,
better `brot`, or better small-`B` training: `chem_lower_bound.py` pins
`sum_rc ||X_rc||_F^2` to `tr(M) = 53.297`, a constant of the FCIDUMP alone, so any
factorization reaching `Lambda ~ 49` has to do it with `H2_I` near zero — and `E_gap`
forbids that, because cancelling `sum_b w_b` leaves the gap 278.8 short. The two
requirements are in direct conflict at this `beta_1`. That is the structural fact the
top of the table must have broken, and it is worth stating plainly: bronze needs a
*new factorization with a different `beta_1` story or a different two-body ansatz*, not
another pass over the existing one.

## The chemistry cost, exactly

```
score   = lambda_eff * T_total * Q_total
T_total = T_angles + T_heur
T_angles = 2 * K * num_vectors,  K = ceil((N-1)/J),  num_vectors = N + R*B
```

With R,B,C = 65,60,30, brot=10, J=10: `num_vectors` = 3,976, K = 8,
T_angles = 63,584, T_heur = 8,794, Q = 731, lambda_eff = 47.85. (The formula
`T_angles = 2*K*num_vectors` is the *unoptimized* QROM count; the actual circuits are
cheaper than it, which is why the measured T_angles is 63,584 rather than 63,616.
The four-address port then takes T_angles to 56,350, measured 65,244/2/25 = 1,305
per circuit.)

## Levers that are now closed

* **J (batching).** J is capped by the 128-qubit limit on each ANGLE circuit:
  `J*brot + selection_bits + 11 <= 128`. At brot=10, sel=12 this gives J <= 10.
  A full scan J=1..12 shows the cost is monotonically decreasing in J and J=10 is
  the last feasible value. **Closed.**
* **brot (angle precision).** Not a knob — a cliff. Re-quantizing the public words
  from 10 to 9 bits takes error_2 from 19.28 to 186.7 mHa (and 238,605 mHa on the
  higher-margin base). The factorization is genuinely a 10-bit object. **Closed.**
* **lambda_eff.** Both accuracy gates sit at the boundary (error_2 = 19.98 of 20 mHa)
  and E_gap is just above its floor. A 1,500-step penalty-method refit of the
  continuous weights found nothing better than the source. **Closed.**
* **C (ensemble width).** Compression 60 -> 30 is accepted (error_2 = 19.98 mHa) and
  is what gives 1.326× over the public 1.318×. Q is flat in C and T_heur falls with
  C, so smaller C helps marginally, but the gain is bounded by T_heur (8.8k of 72k).
  **Now closed by direct measurement:** `drop_copy_probe.py` shows a single dropped copy
  slice takes error_2 to 345.88 mHa, and `drop_rho_row_probe.py` shows a single dropped
  `rho` row takes it to 32.98 mHa. Per-copy mass is near-uniform (1.11 / 1.80 / 2.68
  min/median/max), so there is no slack index. The fit sits at 99.90% of its gate.
* **R (rank).** Same measurement: one row out already fails (19.98 -> 32.98 mHa).
* **`beta_1` / `S`.** Not submitter-controllable — `load_inputs` overwrites both with the
  baseline's `load_fixed_shifts()` before deriving them (`verify.py:218`). **Closed by the
  verifier.**
* **Exact Gram factorization of `T`.** Realizable (unlike the rank-1 SVD idea) and exact
  with `error_2 = 0`, but its nuclear budget is 4.07× the shipped one, giving
  `lambda_eff` 1.96× *worse* even in the best case for `E_gap`. **Closed by measurement.**
* **Selection-register bit width.** Irrelevant: T_angles is linear in the number of
  circuits times the per-circuit load, not in `2^selection_bits`. Halving the row count
  halves the per-circuit work regardless of whether a power-of-two boundary is crossed.

## The `num_vectors` lever: examined and closed

T_angles is ~88% of T_total and is linear in the number of rows loaded. Reaching
1.890× needs the row count roughly halved, i.e. R*B ~ 1,970 instead of 3,900. The
four-address QROM (above) attacks the *per-circuit* constant instead, and gets 1.13x;
this section is about the row count itself.

### Refuted: reducing B below 60 (do not retry)

This is now settled numerically, exactly, not by projection.

`X^(rho c) = U_rho diag(w_c) U_rho^T` is **linear in w**, so for a fixed atom set the best
weights are an exact least-squares solve (verified: it reproduces the source to 6e-15 at
B=60). Using that solve, truncating the atom set is catastrophic and the cliff is at the
very top:

| B | num_vectors | X residual | error_1 | error_2 |
|---:|---:|---:|---:|---:|
| 60 | 3,976 | 6.3e-15 | 4.15 mHa | 19.98 mHa (passes) |
| 58 | 3,846 | 1.25e-01 | 1,614 mHa | 309 mHa |
| 54 | 3,586 | 2.15e-01 | 3,481 mHa | 707 mHa |
| 30 | 2,026 | 5.43e-01 | 11,376 mHa | 3,293 mHa |

The atom basis is rank-exact and non-redundant: all 60 atoms of each row are independent

directions. For the *existing* solution `B >= 60` is forced (see "The structural fact that

governs B" below), so reaching a smaller B requires a **fresh refit of U itself**, not a

selection from the existing atoms.


Three separate measurements along the way were wrong and are recorded so they are not
repeated:

1. A "94% residual-free, 30-atom" claim came from testing *projection* residual inside
   `range(X)`, which is identically zero. Always fit against the target, never against
   its own range.
2. A "union of any row group spans only 30 dims" claim came from `np.hstack` stacking the
   wrong axis, which produced a (2926 x 30) block regardless of group size.
3. `np.linalg.matrix_rank(M)` with the default tolerance normalizes by the largest
   singular value of the *stacked* matrix, so it under-reports rank as the block grows.
   Rank statements must name their tolerance and their scale.

### Refuted: merging rho rows (do not retry)

Merging is **exact** -- done correctly it reproduces error_1, error_2 and Lambda to six
digits (verified at group=2). But it is not a saving: a merged group of g rows needs
`rank = g*C = g*30` atoms while costing `g*B = g*60`, so `num_vectors = N + R*B` is
unchanged (measured: nv 3,976 -> 4,036, slightly worse). `X1 (x) X1 + X2 (x) X2` cannot be
written as a single `X (x) X` term.

### Refuted: C smaller than 30

`rank(X^(rho c)) = 59` for every (rho,c) and `span{X_rc}_c` over 65 rows is 1,950 = R*C,
i.e. maximal. Each (rho,c) pair contributes an independent direction, so C cannot drop
without a refit of U.

**Confirmed directly (2026-10-25)** by `code/drop_copy_probe.py`: dropping one whole copy
slice takes `error_2` from 19.9799 to **345.88 mHa**. Per-copy Frobenius mass is nearly
uniform (min 1.1127, median 1.7978, max 2.6828), so there is no slack copy. The same probe
on whole `rho` rows (`code/drop_rho_row_probe.py`) gives **32.98 mHa** for a single dropped
row. Both fail because the fit sits at 99.90% of the 20 mHa budget. `C` is also the most
valuable knob in the cost model (C 30 -> 16 is -24.9% heuristic `T`), which is exactly why
it is worth having refuted it by measurement rather than argument.

### Refuted: direct SVD of the error_2 target (do not retry)

**Superseded in part by the Gram section above** — the earlier version truncated to rank-1
atoms, which DF-THC cannot realize at all. The *realizable* version (`X_i = sqrt(c_i) *
unvec(phi_i)`, `C = 1`) was measured subsequently and fails for a different reason: the
nuclear budget is 4.07x the shipped one, so `lambda_eff` is 1.96x worse. See "The exact
spectral (Gram) factorization of `T`: refuted, with the arithmetic".

The original note stands as history: the target `T` has a rank-512 SVD residual of only
1.11 mHa with nuclear norm 53.3, projecting lambda_eff 47.54 -> 13.82 (3.44x), but the top
singular vector reshaped to 76x76 has rank-1 fraction 0.024 — it is not `u (x) u`.

### The structural fact that governs B

`X^(rho c) = sum_b w_b^(rho c) u_b u_b^T` is a sum of `B` rank-1 matrices, so
**rank(X^(rho c)) <= B** always. The public solution's `X_rc` have rank 57-60 across all
1,950 pairs. Therefore:

* `B >= 60` is forced for any factorization that keeps the public `X`.
* `B = 30` requires a **different solution** whose `X_rc` all have rank <= 30 (ideally <= 24).
  That is not a truncation, and no amount of weight refitting gets there from the public
  factorization (measured: 51% residual even with optimal atom choice).

There is no rank obstruction from the target: `M = tei + (S(x)I + I(x)S)/2` is numerically
PSD (most negative eigenvalue -3.0e-4, i.e. a 0.298 mHa negative part against a 20 mHa
gate), and a sum of PSD `X(x)X` terms only has to match the PSD part. So a small-B
solution is not excluded a priori -- it just has to be found by training.

### Refuted: training a small-B factorization (do not retry as-is)

Two runs (`code/train_small.py`, B=30 and B=38, seeded from the baseline, 2,000 epochs
each, scored by the official evaluator) both **passed the accuracy gates and failed on
lambda_eff**:

| run | error_1 | error_2 | Lambda | E_gap | lambda_eff | verdict |
|---|---:|---:|---:|---:|---:|---|
| baseline B=60 | 2.4 mHa | 19.3 mHa | 290.0 | 3.92 | **47.54** | passes |
| B=38 ckpt 2000 | 6.35 mHa | 1.68 mHa | 349.8 | 83.2 | 226.5 | fails |
| B=30 ckpt 2000 | 7.19 mHa | 2.05 mHa | 362.2 | 107.0 | 257.0 | fails |

**The accuracy gates are not the obstacle** -- both runs sit comfortably inside 20 mHa, so
B=30 is expressive enough for the two-body term. The blocker is `E_gap`, and it is
structural. With `beta_1` frozen by the verifier,

    E_SOS = ecore - eta*beta_1 - 2*sum|negative eigenvalues of h1_prime| - 0.5*||W||^2

| | sum\|eig\| | tr(h1_prime) | #negative | Lambda | E_gap |
|---|---:|---:|---:|---:|---:|
| baseline | 13.17 | -10.07 | 57 | 290.0 | 3.92 (goal) |
| B=38 ckpt | 112.70 | **-112.70** | 76 | 349.8 | 83.21 |

The trained `h1_prime` is exactly **negative semidefinite**: `tr(h1_prime) = -sum|eig|`
means every eigenvalue is negative, where the baseline is nearly PSD (57 small negatives).
That alone drives E_SOS ~79 Ha too low. Since `lambda_eff = sqrt(E_gap*(2L - E_gap))` and
here E_gap >> L, `lambda_eff ~ E_gap` -- so it lands at 226 instead of 47.5.

Two independent reasons this is not slow convergence but a wrong basin:

1. `tr(h1_prime)` is at its most negative attainable value; the run moved lambda_eff
   250 -> 226 across 2,000 epochs and flattened (500 -> 1000 -> 1990: 253, 253, 257 at
   B=30; 253, 224, 226 at B=38).
2. Lambda moves the *wrong way* as B shrinks. Reaching lambda_eff 47.5 needs
   `Lambda <= ~288` with E_gap at its 3.91 floor. Lambda is 95% the L1 weight norm, and
   fewer atoms must each carry more weight, so small B pushes Lambda up (290 -> 350 -> 362
   as B goes 60 -> 38 -> 30).

The one untried fix is a **PSD-preserving initialization** for `h1_prime` (seed from the
baseline's near-PSD structure instead of letting the truncated-B start flip the sign). That
is a real hypothesis, not a projection, but it does not address reason 2, and Lambda being
monotone in B suggests the trade-off cannot reach 288 at B=30 even if E_gap is repaired.

Projected payoff, if a small-B factorization with repaired E_gap could be found:

| R,B,C | num_vectors | T | Q | chemistry | composite |
|---|---:|---:|---:|---:|---:|
| 65,60,30 (now) | 3,976 | 72,410 | 731 | 1.326× | 5.553 |
| 65,30,30 | 2,026 | 40,673 | 696 | 2.478× | 6.840 |
| 65,24,60 | 1,636 | 38,704 | 699 | 2.593× | 6.944 |
| 65,20,60 | 1,376 | 34,480 | 699 | 2.910× | 7.217 |

**None of these are achieved.** They assume both the accuracy gates *and* the E_gap floor
hold, and the runs above show the E_gap floor is what fails.

## Tools (all under `code/`)

| file | purpose |
|---|---|
| `env.py` | TMP/TEMP + sys.path setup (kickmix's `tmpfile()` needs a writable temp dir) |
| `run_invsqrt.py`, `run_dqi.py`, `score.py` | drive the official verifiers |
| `chem_refit.py` | torch port of the verifier's metrics + penalty-method refit of (w, W) |
| `chem_rotate.py` | per-rho orthogonal c-rotation to minimize Lambda's weight term |
| `chem_compress.py` | ensemble compression (W^T G W diagonalization + Givens L1 rotations) |
| `select_atoms.py` | exact least-squares refit of w at a truncated atom set (proves B=60 is tight) |
| `build_seed.py` | best-atom-subset seed for a smaller B (greedy selection + exact refit) |
| `train_small.py` | background small-B trainer, checkpoints scored by the official evaluator |
| `offset_opt.py` | L-BFGS-B on lambda_eff with the exact eigh gradient (ports the public 6.69 method) |
| `four_addr/banked_qrom.py` | four-address banked QROM + signed Givens gauge, re-implemented from scratch (see the port section) |
| `four_addr/build.py` | builds the ANGLE family for a chosen (s, J) and scores it with the official verifier |
| `four_addr/design_sweep.py` | exhaustive (s, J) sweep of the family under the 128-qubit cap |
| `four_addr/rank_compress.py` | GF(2) rank of the difference bank — proves it is incompressible |
| `four_addr/bit9_redundancy.py` | checks the gauge's `bit9 == bit8` identity and the wire saving it implies |
| `four_addr/test_circuit.py` | semantic per-address XOR test of one ANGLE circuit in the simulator |
| `four_addr/package.py` | assembles `submission.zip` and checks member order, CRCs and per-file hashes |
| `four_addr/verify_archive.py` | unpacks the zip and re-runs all three official verifiers end to end |
| `brot7_probe.py` | requantises the shipped angle words to each width and scores them with the verifier's `evaluate_words` |
| `chem_ceiling.py` | the (B, brot) cost-model ceiling, including the `brot <= 7` temporary-qubit cliff |
| `lambda_breakdown.py` | decomposes `Lambda` into `lambda_h1` and `lambda_h2`, and prices the `W` freedom |
| `W_freedom_probe.py` | evaluates `Lambda` / `E_gap` / `lambda_eff` for a family of submitted `H2_I` values |
| `h1p_null.py` | least-squares solve for the `W` that nulls `h1_prime`, and the resulting `Lambda` |
| `opt_W.py`, `opt_W2.py` | constrained optimisation of `W` against `Lambda` and the `E_gap` floor |
| `invsqrt90/build_invsqrt90.py` | the four-stage 90-qubit build, asserting each intermediate digest |
| `invsqrt90/{signed_digit_invsqrt_v3,selected_schedule90,fusion_schedule90,…}.py` | the ported source-exact modules (Apache-2.0) |
| `dqi_src/build_dqi.py` | the source-built BM decoder driver |
| `dqi_src/{dqi_driver,gen_dqi_v6/*}.py` | the ported dqi source chain (Apache-2.0) |
| `train_rank.py` | official DF-THC trainer at a chosen (rank, bases, copies) |
| `chem_W_opt.py` | autograd over all 1,950 `W` entries to minimise `lambda_eff` with every gate enforced |
| `chem_lower_bound.py` | the `tr(M)` / nuclear-norm floors on `lambda_h2`, and the `E_gap` coupling |
| `chem_hard_floor.py` | the `E_gap`-implied floor `Lambda >= p + 0.25*(sqrt(1126.09 - 4p) - ||s||)^2` |
| `rank_probe.py` | `error_2` vs a per-`X_rc` rank cap — the ceiling a `B = k` factorization could reach |
| `rank_per_rc.py` | per-entry rank distribution and mass-beyond-k; why "mass kept" is the wrong metric |
| `drop_rho_probe.py` | entry-level `error_2` probe (zeroes `(rho,c)` entries; changes no cost) |
| `drop_rho_row_probe.py` | drops whole `rho` rows — the cost-bearing R test (1 row already fails) |
| `drop_copy_probe.py` | drops whole `C` copy slices — the cost-bearing C test (1 slice already fails) |
| `rbc_marginal.py` | d(heuristic `T`) and d(`Q`) for perturbations of `(R,B,C)`, from `SpinFreeData_V1` |
| `m_spec_sqrt.py` | exact Gram/eigen factorization of `T`; spectrum and truncation frontier |
| `spectral_budget.py` | the `lambda_h2` nuclear budget of that factorization (4.07x shipped) |
| `lambda_egap_identity.py` | measures that `E_gap`, not `Lambda`, is the binding gate |
| `lambda_floor_hard.py` | the `E_gap`-forced `Lambda(p)` curve and its minimiser |
| `dqi_src/peak_report.py` | live `Q` profile by phase for the dqi build; `--compact-from` / `--stack` |
| `prof2.py` | per-stage Toffoli accounting of the reference invsqrt circuit |
| `chem_gauge_lib.py` | bit-exact local reimplementation of `evaluate_dfthc_angle_words`, validated against the official kernel |
| `chem_gauge_opt.py` | exact-metric LP over all 488 `w`/`w_B` variables with analytic λ/e₂/E_gap gradients — proves λ is closed at −2.5e-8 |
| `chem_gauge_scan.py` | the nonlocal 2-parameter basin scan (441 points, exactly one feasible) |
| `angle_costs.py` | static CCX counting for a `.kmb`, matching the verifier's 2× charge |
| `batch_table.py` | the per-batch CCX vector of a bundle, and the elementwise min across bundles |
| `peephole_scan.py` | adjacent-duplicate / within-4 / repeated-qubit Toffoli scan (empty on all 15) |
| `dl_chem_batches.py` | fetches full 15-circuit sets for the bundles worth comparing |
| `inv_stats.py`, `inv_live.py`, `inv_peak.py` | invsqrt qubit-liveness profile, peak width, and the 63-qubit proof |
| `inv_cswap.py`, `inv_cswap2.py` | CSWAP minimality and the compute/uncompute triple census |
| `inv_worst.py`, `inv_value.py` | the 6.3M-shot deterministic worst point and the per-lever exchange rates |
| `inv_manifest.py`, `inv_manifest2.py` | the mrmeows source release: generator, negative-results README, constant-seed model |

## Constant-seed model reconstruction (2026-10-30, numerics only — no circuit, no submission)

The `research/` directory of `invsqrt_19p80_history_cache_research.zip` does not exist in the
dataset tree (no entry in `SHA256SUMS`), so mrmeows' constant-seed model was rebuilt from the
README spec: **128-entry seed table, 8 fractional bits, degree 3, residual 24+2 guard bits,
output fraction 26, worst error/tolerance 0.617 over the full domain**. Code and artifacts in
`_scratch/invsqrt/constseed/` (`model.py`, `sweep.py`, `certify.py`, `sweep_out.txt`,
`certify_out.txt`, `constseed_seeds.{npz,txt}`, `ref.npz`).

### Reconstructed model family

With `e = bitlen(x)-1`, `u = x/2^e ∈ [1,2)` (exact: `x` has ≤ 22 bits), `b` = top 7 bits of the
mantissa fraction of `u`:

- `s_b` = constant seed, 8 fractional bits, quantised `1/sqrt(u_c)` at the bucket anchor;
- `m ← s(3-u·s²)/2` applied twice — two Newton refinements of a degree-0 seed give a
  **degree-3 polynomial in `u`** with all coefficients derived from the stored constant;
- residual channel: `u-1` at 24+2 guard fractional bits (exact here — the domain never needs
  more than 21, so the guard budget is pure slack);
- output `k` = `m` quantised to 26 fraction bits; `theta = k·2^(-26-e/2)` turns. Kickback is
  linear in the output bits (bit `j` is a phase control of weight `2^(-j-26-e/2)`, odd `e`
  folds `1/sqrt(2)` into constant angles): **`1/sqrt(x)` is never stored in full**, and the
  normalization + Newton–Raphson block is gone — `e` is a wire reindexing, not arithmetic.

Semantics are exactly the verifier's: `target = round(isqrt(2^130/x)/2) mod 2^64`,
`tol = isqrt(2^130/x) >> 25`, PASS iff `|theta·2^64 - target| ≤ tol` (int64 view).

### Full-domain certification (all 3,133,452 legal x)

| variant | worst err/tol | worst x | min margin (2^-64 turns) |
|---|---:|---:|---:|
| center anchor, round output | **0.183370** | 2,088,791 | 524,955,476 |
| center anchor, floor output | 0.443215 | 2,048,087 | 396,802,383 |
| left anchor, round output | 0.196180 | 1,023,950 | 516,698,858 |
| left anchor, floor output | 0.382134 | 2,031,675 | 428,919,989 |

All four ≤ 0.617. The top-25 worst points were re-verified in exact `Fraction` arithmetic
(dyadic model; `sqrt(2)` bounded by a 200-bit integer isqrt for odd `e`) — identical ratios.
Per-bucket seed optimisation (floor/round/ceil ±1 ulp candidates) does **not** move the worst
ratio below 0.183370: the binding constraint is the 26-bit output quantisation
(bound `0.25/sqrt(2) = 0.17678` plus the degree-3 residual), not the seeds. Compare the
shipped architecture's 0.99932 — the model leaves ~5.4x more phase margin (5.25e8 vs
5.86e7 units), which is the asset the README describes.

### Where 0.617 sits (and the parameter sweep, 480 specs, full domain each)

- **Degree 1** (one Newton step) fails at every seed width (worst 6e3–5e4). **Degree 3** is the
  minimum: with 128 buckets it needs ≥ 7 seed fractional bits (7 → 0.328, 8 → 0.183, 9 → 0.177).
  **Degree 5** passes even with 5-bit seeds. Output fraction 25 (round) doubles the ratio to
  0.353; floor mode doubles it again — all consistent with the quantisation bound
  `2^(24-of)·0.5/m_min`.
- With circuit-realistic **product truncation** inside the Newton steps (`pt` fractional bits
  kept per multiply), the nominal spec gives: `pt=24` → FAIL (1.02–1.46); `pt=25` → 0.476–0.780;
  `pt=26` → 0.241–0.433; `w`-only truncation at 24 → exactly 0.500. **"Residual 24 + 2 guard
  bits" is precisely the fail/pass boundary of this family** — 24 value bits fail, 24+2 passes.
  mrmeows' 0.617 is bracketed by our variants (0.500 ↔ 0.71–0.78) and is certainly one specific
  truncated-arithmetic point of the same family; the zip that would pin the byte-exact variant
  is missing, but every clean implementation of the nominal spec passes at ≤ 0.617 with room,
  which is the claim that matters.

### If compiled: what the replacement block costs (qualitative)

Removed: the normalization produce-and-decode chain (front decoder reuse, 10.5–38 Toffolis of
mean-cost decoders per the README lineage) and the NR inverse arithmetic with its derivative-seed
machinery. Added: a 1,024-bit constant QROM (7-bit address), one 8x8 square (~56), two truncated
multiplies by the shared multiplicand `u` (`26x16` and `26x26` kept to 26-27 columns) and one
8x27 multiply for `m1` — naive schoolbook ~1.6-1.9k Toffolis, i.e. **over** the ~690+52 budget;
with column truncation (certified at `pt=26`, ratio ≤ 0.43), `u`-multiplicand sharing and
Karatsuba-style splitting, a realistic target is ~700-1,100 — borderline at the stated budget,
and the certified margin (0.183 vs the 0.617 budget) is what buys the aggressive truncation.
Rotation count drops from 41 taps to ~27-28 output-bit phase classes, and the 8.96x margin
gain lowers per-rotation synthesis cost further (~0.9x T total at equal eps split). Net:
the model is numerically sound and worth compiling only if the shared-product machinery from
the dfthcsos lookup compiler brings the evaluator under ~1,100 Toffoli-equivalents.

Verdict: **claim verified as stated** — the constant-seed spec passes the full-domain numerical
check with worst error/tolerance ≤ 0.617 (our cleanest reconstruction: 0.183), the margin is
real, and it has not been compiled by anyone; the compile is the open lever.

## The nine pole flags are live data: why heliosli's 889 does not port as an edit (2026-10-30, refuted)

Attempted: Q 897 → 889 by deleting the nine `pole` zero flags of `external_node_chain` (the
only 8-qubit lever in the tree, per "Where the 57 scratch actually are" above). Result:
**not reachable by substitution.** Four builds and three official verifier runs; the mechanism
is now pinned, and it is *not* the one recorded earlier ("the flag is a computational-basis
value, but after `cswap(pole, N, D)` it is entangled with N and D, so measuring mid-circuit
collapses the state").

### 1. The ISA cannot read a qubit's value (kickmix source, `circuit/op_type.h`, `sim/sim.h`)

There is no Z-basis measurement anywhere in the instruction set. The only two ways to release a
qubit are

- `HMR q b` = Hadamard + measure + reset. Simulator (`sim.h:405`):
  `b ^= rng & cond; phase ^= q & b & cond; q := 0`. The outcome bit is **uniformly random,
  independent of q**; the value q is destroyed and reappears only as a phase kickback `q&b`.
- `R q` = "ensures a qubit is 0 (randomizes global phase if qubit was 1)": `phase ^= q & rng`.

So a qubit value can never be moved into a classical bit, and releasing a flag always costs a
random phase that must be *repaid by re-deriving the same predicate*. That is exactly the
`hold_flag` / `consume_flag` idiom of `final_decoder/conversion.py`, and it is why
`consume_flag` is always paired with `fresh_zero_flag(register)` at reverse time: the deferral
is only sound when the predicate can be recomputed from a register that is restored *before*
the flag is needed. The shipped 897 stream is a per-shot (basis state, global phase) computation
— there is no entanglement to collapse, so the earlier explanation was wrong in mechanism.

### 2. Measured: the nine flags fire on a large fraction of shots

Probe build (`DQI_ZTC_PROBE=1`): one `Z pole` gate immediately after each `zero_flag`, circuit
otherwise byte-for-byte the shipped one. A Z on the flag negates the global phase iff the flag
is 1, so a firing flag can only show up as a *phase* error:

```
{"sha256":"ebdcf740a6e440bf…","Q":897,"T_static":444667,"num_ops":59852050}
FAIL  value_errors 0/102400   phase_errors 21676/102400 (21.2%)
```

21.2% of shots carry an odd number of set flags. The flags are load-bearing data, not a
formality — this alone rules out "measure it and re-prepare" (measuring injects
`phase ^= flag & random`, unrepayable without the predicate).

### 3. Three reconstructions of the pole at reverse entry, all rejected

Each row is one build plus one full 102,400-shot official verifier run.

| variant | forward | reverse entry | Q | T_static | verifier |
|---|---|---|---:|---:|---|
| ZTC-A `3ef484a5…` (earlier) | `HMR` flag | re-prepare from the HMR bit | 889 | 444,667 | FAIL 0.9997 (102,220 value / 46 phase) |
| ZTC-B `f6b260ac…` | `HMR` flag | fresh qubit left at \|0> (branch hard-wired) | 889 | 444,667 | FAIL 32,812 value / 1 phase |
| ZTC-C `c8c093c4…` | `HMR` flag, repaid | `zero_flag(N)` = `[N=0]` as the pole | 889 | 444,874 | FAIL 23,033 value / 0 phase |

- **ZTC-B** isolates the branch: with the reverse conditional swap never taken, 32% of shots
  are wrong. So the reverse consumption is real work, not bookkeeping.
- **ZTC-C** isolates the predicate. The reverse entry only has N, and `[N=0]` is *not*
  `[D=0]`: N at that moment is the node's word **after** its normalize and after the
  extraction-level scramble of `external_extraction`, not the post-swap N. Where they disagree
  the branch is wrong; the `CZ` repayment of the HMR kickback fails on exactly those shots,
  which is why the phase count is 0 (masked: the verifier scores `phase_err & ~val_err`).
  This variant is the framework's own deferral idiom applied as faithfully as possible, and it
  is the closest reconstruction of a "zero-test chain" — it still fails.

All three reproduce to Q = 889 with T within +207 static of the 444,667 baseline; had one
passed it would have scored 4.0556e11 → composite **22.9189** (silver with margin). The width
was never the problem.

### 4. Why heliosli's 889 exists anyway (from their bytes, not from us)

- Their nine insertions are exactly one `HMR pole` after each of the nine pole CSWAP groups in
  the forward extract (ops 4,387,064 / 4,626,923 / 4,890,179 / 5,177,304 / 5,488,316 /
  5,822,979 / 6,181,111 / 6,562,910 / 6,968,550), one per homogeneous node, each into its own
  dedicated bit (`b101229, b101325, b101386, b101447, b101508, b101569, b101630, b101691,
  b101752`). The tenth, extraction-level pole (control `q857`) is untouched. So their forward is
  "create the flag, swap with it, then measure it away".
- Those nine bits are *consumed* later — 142 references each, last use a `PUSH_CONDITION` at the
  matching node's reverse (53.70M, 53.27M, 52.78M, 52.24M, 51.65M, 51.01M, 50.32M, 49.58M,
  48.78M, i.e. innermost node first). So their construction is not "measure and forget": the
  random bit is replayed through conditional phase/arithmetic networks (`BIT_INVERT … if b`,
  `CZ … if b`, `push_condition(b); CX/CCZ …; pop_condition`) against the registers that
  produced the product masks — the `phase_product24`-style repayment, driven by a bit that is
  *expected* to be random.
- But their 24-CSWAP groups in that same reverse are **not** conditioned: they are plain
  `CSWAP` under a single control, and that control is not one of their forward pole qubits.
  Forward poles: `{q886, q861, q844, q852, q855, q849, q878, q864, q853}`; reverse controls (3
  runs per node): `q857, q843, q858, q855, q885, q840, q879, q843, q858, q855` — only seven
  distinct qubits for ten nodes, so the pole is re-allocated per node, and there is no write to
  it anywhere in between (`q843`: none in `[7.0M, 48.17M]`; `q857`: none in `[7.3M, 47.84M]`).
  A re-allocated qubit is |0>, so **the reverse conditional swap in their circuit never fires**:
  they deleted the data branch and kept only the phase replay.
- Hence their 889 is sound only on inputs where `[D=0]` never fires at a homogeneous node.
  Their file is a different assembly from ours: 55,833,283 ops (`b3354f60` lineage, the base
  they forked) against 59,852,041 in the tree we rebuild (`53d8b2ed`, v19). In v19 the guard
  fires (21.2% odd parity, 32% at ≥ one node), so the same edit cannot be ported: copying their
  op pattern *is* ZTC-B, and ZTC-B is 32% wrong. The 993-op diff that the earlier section quotes
  is `b3354f60` ↔ heliosli, not v19 ↔ heliosli.

Verdict: **the pole route to Q ≤ 895 is closed on this tree.** Each of the nine flags is a data
control that (a) is consumed before anything it depends on is restored and (b) cannot be
converted to classical control in this ISA. What would be needed is an algebraic restructure of
the safe-scalar guard so the degenerate branch is decided from restored data (the reverse
re-deriving the branch instead of being told it) — a rewrite of
`external_node_chain`/`external_normalize_paired`, not a substitution. Do not retry
measure/re-prepare, hard-wire-0, or `[N=0]`.

### 5. The remaining single lever to silver is T̄, not Q

With R already at the 0-error floor (1.1912604648) and Q fixed at 897, the cost is
4.129431e11 against a silver line of 4.119759e11, i.e. **T̄ must drop 0.234% = 1,010 mean
Toffoli** (430,822.9 → 429,813). That is a ~0.23% kernel trim inside `scan`/`convert`, ~50x
smaller than the 8-qubit prize that is now closed. Nothing else in the model moves: eps is
already 0/102,400, Q is at its true 897 minimum by interval colouring, and the only Q lever
(the flags) is refuted above.

Artefacts: builds `_scratch/dqi_ztc/{probe_f,ztc2,ztc3,repro_default}.kmb` (+ `.build.json`),
logs `_scratch/dqi_ztc/{build_probe,verify_probe,build_ztc2,verify_ztc2,build_ztc3,verify_ztc3,
build_repro}.log`, heliosli/base op streams `_scratch/dqi_ztc/{helio,base}.kmx`. Source changes
are behind `DQI_ZTC` / `DQI_ZTC_PROBE` (both default 0): the default build still reproduces
`53d8b2edf251e2a0…`, Q 897, T_static 444,667, num_ops 59,852,041, verified again today.


## The `exp/` cluster, audited: every "cheaper division" is dominated inside the absorb reverse (2026-10-30, refuted by measurement)

Ask: shave T̄ by 1,010 (0.234%) to reach silver while Q stays 897. The 23 directories under
`playbook/kit/decoder/exp/` and the 12 files under `exp/r-main/stream/` are the same authors'
cost/width playground; several carry explicit Toffoli counts. All were read (docstrings, kernel
comments, call sites); the full table is in `_scratch/dqi_tbar/audit.md`. Headline: **the live chain
already holds the cheapest variant of every kernel class**, and the one class with real savings on
paper (the three families of "369/414-T" zero-safe divisions) is dominated *at this call site* by
the fused division it was meant to replace.

### Where the T actually is (executed, not static)

The verifier's mean is 430,822.88 against T_static 444,667. The 13,844 gap is branch bodies of
12-way classical-mask phase kernels (`phase_inverse36_pivot`, `phase_inverse6_classical6`), of which
one branch is taken per shot. **A static drop inside those kernels buys nothing.** The mean
decomposes exactly as CCX 416,153 + CCZ 726 + CSWAP 13,944 per shot (CSWAP alone is 3.24%).

Region map from a class-level instrumented rebuild (`_scratch/dqi_tbar/build_instr3.py`, emitted
counts, reproduces Q 897 / T 444,667 / ops 59,852,041 exactly): `convert_safe` 113,740;
`external_chain_step` 79,002; `divide_fused` 48,810 (33 calls, including its after callback);
`scan` 44,275; `external_scalar_step` 40,095; `synthesis` 32,220; `external_normalize_paired` 9,288.
(The proxy-object instrumenting approach is unusable here: pybind rejects a `stride_span` third
operand only when a proxy frame sits in the call path, and the build dies inside `inverse6_xor19`.)

### Measured micro-costs of every division on offer (isolated 24+24+1-qubit circuit)

| kernel | emitted T | private Q | note |
|---|---:|---:|---|
| `divide_fused` (live, 33×) | 704 | 30 | half of it is branch bodies; **executes ≈327** |
| `divide_flag369` (this round's swap) | ~524 + 23 `phase_zero` | 25 | branch-free, executes ≈392 |
| `fold_joint.divide_zero371` | 503 | 37 | own zero flag |
| `tail_alias.divide_zero369` | 501 | 37 | 66-T inverse |
| `mul99.divide_lowwidth369` | 501 | 24 | 12-branch phase |
| `mul99.divide_lowwidth372` | 372 | 24 | 39-T phase |
| `fold_cached_norm.divide323` | 323 | 36 | unit factor, needs y≠0 |
| `arithmetic.divide_via_norm` | 321 | 43 | widest |

### The round that was actually built and verified

`absorb_odd_fused` / `absorb_even_fused` reverse were routed through `_absorb_divide`; under
`DQI_ABSORB_DIV=1` it runs `divide_flag369` (the old `divide414_callback` body minus its second zero
test, consuming the caller's live flag), otherwise the shipped `divide_fused`. The default build is
byte-identical after the edit: `53d8b2edf251e2a0…`, Q 897, 444,667, 59,852,041.

| | baseline | `DQI_ABSORB_DIV=1` |
|---|---:|---:|
| sha | `53d8b2ed…` | `d2441290…` |
| Q | 897 | 897 |
| T_static / num_ops | 444,667 / 59,852,041 | 438,694 / 59,603,848 |
| verifier | PASS 0/0, R 1.1912604648 | **PASS 0/0, R 1.1912604648** |
| mean Toffoli | 430,822.88 | **432,970.71 (+2,147.8)** |
| cost | 4.129431e11 | 4.150019e11 |
| composite | 22.78244 | 22.7446 |

Functionally flawless (0 value, 0 phase, same R, same Q) and still worse: 33 folds × (392 − 327) ≈
+2,145 ≈ the observed +2,148. So `divide_fused`'s 704 static is really ≈327 executed — it already
*is* the 323-class kernel, bought with classical-mask phase fusion instead of an unconditional
`phase_zero` (22 ANDs + 1 CCZ) — and every 369/414/371 variant adds ~65 executed T per fold here.
The same verdict covers the other audited near-misses: `inverse_width` reverse 220 > live 209,
`kernel19` reverse 219 > 209, `jet24/kernel` 686 > `pro_jet` 658, `phase39` 39 > `phase36` 36,
`phase24_tradeoff` 118 > the jet's 106, `safe_scalar.inverse` 67 > `zero1` 65. Nothing in `exp/` is
both unwired and cheaper in the executed metric.

### What this closes, and the one shape of lever that remains

Do not retry: swapping any absorb-reverse division to 369/414/371/323/321 (measured), any jet /
phase / I24 / I12 kernel variant from `exp/` (dominated), or reasoning "static T fell, so mean will".

The remaining ≥1,000-mean levers are structural, not substitutions: (a) fewer than 33 absorb folds
or fewer than 17 converter levels (algebra), or (b) the 13,944 CSWAP/shot converter route — 3.2% of
the mean, currently a word-wise `cswap` chain under a live quantum flag. Either is a rewrite of
`converter`/`synthesis`, not a wiring job.

Artefacts: `_scratch/dqi_tbar/{audit.md, baseline_check.kmb, regress_default.kmb, v369.kmb,
verify_baseline.log, verify_v369.log, instr_report3.json, micro_divide.json, micro_divide_exec.json,
build_instr2.py, build_instr3.py, micro_divide.py, micro_divide_exec.py}`. Source change: one
env-guarded switch in `playbook/kit/decoder/final_decoder/synthesis.py` (`DQI_ABSORB_DIV`, default
`0`); no shipped-default byte changed, nothing submitted.

## invsqrt 62q v44: the three shrinking routes, measured shut (2026-10-30, refuted)

Target was ≤100,520 (−0.48%). Baseline `heliosli-v44` (`a568bc79…`) verified PASS at
101,002.4784 (Q 62, TE 1,629.0722, toffoli 1,332.4816, 41 rotations at eps 2.434e-13,
t_count 6.999, 805 depth, 47,190 ops, worst error/tol 0.99932377 at x=161). **No candidate
was found.** The baseline itself is the archived best (`_scratch/inv_shrink/`). The exchange
rate makes the arithmetic unforgiving: one CCX/CCZ/CSWAP = 1 TE = **62 score**; one non-T
rotation = 7.234 TE = **448.5 score**. Silver needs −7.78 TE; scramble needs −23.30 TE.

### Route 1, Q 62→61: closed by conflict, not by inspection

Live-qubit profile is a flat 62 from op 3,773 to 36,787 (33,015 ops long), with five more
62-peak windows; segmenting each qubit at every HMR/R (1,037 segments) leaves the segment-level
maximum at 62. All 780 scratch pairs collide at least once (the best pair collides 1–2 times),
and a nested-lifetime search inside HMR idle windows returns 0 candidates; inside the peak
windows every qubit is touched 200–1,400+ times. Getting 61 means re-scheduling the main
multiplier's scratch allocation — a recompile, not a renumbering.

### Route 2, static redundancy: five sub-classes, all empty

| class | scan | result |
|---|---|---|
| adjacent identical / adjacent inverse pairs | `inv_dup.py` | 0 / 0 |
| "clean compute–uncompute pairs" | `inv_dup.py` | 80 flagged, but they are interleaved MX CSWAPs around `Z_POW q28`+`HMR q28` — convolution, deleting changes semantics |
| degenerate CCZ (c1∧c2 ⇒ c3) | `inv_cczscan.py` | 41 flagged, all low-sample artefacts; live CCZs all fire |
| control constantly 1 (→ free CX/CZ/SWAP) | `inv_ctrlscan.py`, 255k x × 3 seeds | **0 gates** |
| control constantly 0 (→ delete) | same | 0.0225 TE total, largest item 0.0077 |
| gate never fires (→ delete) | `inv_deadscan.py`, 255k x × 3 seeds | 41 gates, all inside condition blocks (max exec 6.2%) → **≈0.15 TE** |

`inv_deadscan` also reproduces the official 1,332.4816 mean from raw state simulation
(1,332.484), which is what licenses the whole table.

### Route 3, 41→40 rotations: the angles are already max–min optimal

The earlier "already converged" LP used the wrong tap model — a `Z_POW`'s phase rides on the
**target qubit's value at execution time**, not on the HMR bit measured afterwards (HMR is an
X-basis random outcome). Re-extracting the true V columns (`inv_tapval.py`, 41 taps → 42
prefix chunks, 37 s) and re-running max–min:

* boxes ±100% / ±200% / ±400%: converged at round 0, margin 58,597,850 → 58,597,850;
* fully free per angle in [−0.5, 0.5] turn: `LP m = 58,597,853`, i.e. **1.000×**, all moves < 1e−7.

Zero is inside every box, so "just drop a tap" is infeasible too, and the PASS slack is only
0.068% (0.99932 of tolerance). V columns have no duplicate and no complementary pair, and a
single-tap scale sweep over ±1.2× finds nothing.

### Route 4 (new): conditioning a Toffoli on a bit that implies it fires — closed

If `fire ⊆ {b = 1}`, then `op … if b` is the identity whenever b = 0: semantics untouched,
cost no longer charged on those shots. `inv_apply_cond.py` builds it and re-simulates to
confirm a bit-identical final state (0 qubit mismatches). Varying the evidence threshold:

| min fires observed per bit | gates surviving | nominal saving |
|---:|---:|---:|
| 0 | 15 | 9.01 TE |
| 2 | 7 | 2.24 TE |
| 5 | 1 | 0.15 TE |
| 10 / 20 / 40 | **0** | **0** |

At 890,481 x samples with a single bit and no threshold, the survivors fire **8, 3, 2, 2, 2**
times out of 890,481 — the multiple-comparison residue of ~500 candidate bits × 1,502 gates,
not structure. **The earlier "9 gates / 4.43 TE" estimate was this same artefact and is
withdrawn.**

### What that leaves

Everything reachable by selection, renumbering or local rewriting is worth **0.17 TE (≈10
score, 0.01%)**. The remaining lever has to change tap count: one rotation is 448.5 score, so
39–40 taps with ≤1–2 extra Toffolis lands at 100,106–100,554, i.e. silver and near the
scramble line. That is the reverse-residual re-synthesis (taps 15–40: 26 rotations on q28 plus
26 two-controlled multiplications) — the same job shape as the unbuilt constant-seed compiler,
and the only invsqrt lever left.

Artefacts: `_scratch/inv_shrink/{invsqrt_base_best.kmb, verify_invsqrt_base_best.json,
verify_base_best.log, NOTES.md, csim_report.npz, fit_*.npz}`; tools
`_tools/kprobe/{inv_live,inv_seg,inv_pair,inv_clique,inv_win,inv_dup,inv_cczscan,inv_tapval,
inv_fit2,inv_fit3,inv_apply_cond,inv_deadscan,inv_ctrlscan}.py`. Nothing submitted; the
harvest bundle and `solutions/vqcc/_scratch/submission.zip` were not touched.

## The chemistry squeeze on the heliosli v44 corner: both levers measured closed (2026-10-30)

Goal: beat the just-harvested chemistry cost 337,993,966.88 (λ_eff 48.323649, T 16,266,
Q 430; `_scratch/harvest1030/verify_chem_v44.json`, reproduced locally to the digit:
`_scratch/chem_squeeze/logs/verify_base.log`). Target −0.48 % (≤ 336.37e6), stretch −1.0 %
(≤ 334.6e6); at the measured composite elasticity (+0.323 per −1.508 % chem cost, the
10-29→10-30 swap), −0.48 % is worth ≈ +0.103 composite (22.996 → 23.098, just under silver).

### λ route: the shipped (w, W) is the fixed-word optimum — corner re-proven on the v23 words

The v44 metrics sit on a three-gate corner, tighter than the old family's:
`error_1 19.99941 mHa, error_2 19.99944 mHa` (cap 20), `E_gap = 3.9148990000 = floor +
1.0e-6` (strict floor 3.914898), and `||W||² = 1080.1768` equals the E_gap-implied minimum
`1126.0947 − 4p` exactly (`chem_hard_floor.py`, p = 11.4795, ||s|| = 22.846). Λ = 300.1995
= λ_h1 12.96 + λ_h2 287.24.

* `chem_refit.py` with its own default margins (errors ≤ 19.98 mHa, gap ≥ floor + 5e-5):
  **no feasible iterate found** in 6,000 steps — the shipped corner is tighter than the
  refitter's standard slack (`logs/refit_v1.log`).
* Refit with margins bracketing the corner (errors ≤ 19.999 mHa, gap ≥ floor + 3e-7,
  lr 2e-4·rms, 2,500 steps, NaN-guarded): the start is the best feasible iterate at every
  checkpoint; every λ-descent direction walks E_gap below the floor (step 2500: λ 48.2949
  but gap 3.9105, infeasible). Official re-check of the best iterate = the shipped numbers
  exactly (`logs/refit_v2.log`).
* This closes the "refit for the new v23 words" question the 10-29 LP could not answer:
  the corner structure is not an artefact of the old family — heliosli (or their
  predecessor) already shipped the fixed-word (w, W) optimum, with both error gates and
  E_gap simultaneously saturated.

**λ headroom on the v44 words: 0.** The forum's 46.8 "hard floor" would require new
vectors (a re-trained factorization), not new weights.

### T route: v23 is already at the measurement-uncompute optimum; the dictionary is the only lever left

The 15 circuits were decoded gate-by-gate (`v23_probe.py`, `dag_recover.py`,
`hmr_gain.py` in `_scratch/chem_squeeze/`):

* Static per circuit: 258–301 CCX (Σ 4,168; billed 2× = 8,336), 80 HMR + 80 CZ, everything
  else CX (unbilled). Peephole empty on all 15 (0 adjacent/commuting duplicates, 0
  degenerate gates).
* The structure is the somejoint plan **plus** full measurement-based uncomputation: 75
  workspace ancillas compute shared products (GF(2) products of selection-bit monomials,
  fan-out to up to ~160 word bits via free CX), and **all** of them are HMR-uncomputed
  (the 80 HMR include 5 mid-circuit slot reuses). The remaining 183–226 CCX write
  single-use products directly into word bits. There is **no CCX-reversal left to
  convert** — the "replace reversal CCX by HMR + conditioned CZ" gain this session went
  looking for is already fully banked by heliosli. (The HMR + `CZ c1 c2 if b` correction
  is exact for arbitrary control values: CZ applies (−1)^{v(c1)·v(c2)} = the measured
  X-basis phase exactly.)
* Word tables are XOR-differenced per batch (`delta[k] = batch[k] ⊕ batch[k−1]`), so the
  "the 15 circuits share one angle set" premise is false: each circuit's billed CCX is the
  multiplicative complexity of its own 40-bit delta function, and cross-circuit sharing
  cannot reduce per-circuit execution anyway (circuits run sequentially and workspace must
  be clean after each).
* The delta ANF is dense (2,036 degree-≥2 monomials per circuit over 11 selection bits);
  the shipped circuits compute it with 258–301 — the dictionary is doing real work. Within
  each circuit: **0 duplicate products** (across the 75 shared + 183–226 direct writes, no
  product is computed twice, none is both shared and direct), so the plan is
  irredundant at the product level too.
* Q = 430 is shape-fixed (R8/B143/C60/brot 8) and every measured cheaper-Q option pays
  ≥ +12 % T (10-29 section); brot ≤ 7 stays dead (needs re-encoded words; gates are
  saturated); J is forced by `len(circuits) = ceil(75/J)` with the 128-qubit cap leaving
  only 2 spare qubits.

**Remaining T reduction = a strictly better joint block dictionary** (−20 products ≈
−0.48 %): the same open problem as before, now against heliosli's ~2,968-product family
instead of somejoint's irredundant 3,018. No public construction exists; none found here.

### Verdict

No candidate beats 337,993,966.88 this session; nothing submitted, nothing packed beyond
the verified repack script. Deliverables in `_scratch/chem_squeeze/`: `base/` (the
reproduced submission + `verify.json`), `logs/` (refit v1/v2, local baseline verify),
analysis scripts (`v23_probe.py`, `anf_analysis.py`, `dag_recover.py`, `hmr_gain.py`,
`hmr_transform.py`, `refit_v2.py`) and `pack_reconstruct.py` (reproduces the harvest1030
zip byte-for-byte, checked against all 16 dfthcsos members). Composite unchanged at
22.99562; silver (23.10513) needs chemistry −0.52 % or an equivalent gain elsewhere —
both chemistry levers are now measured shut at this corner, so the composite can only move
via the invsqrt tap re-synthesis (−39/40 taps) or a new public chemistry artifact.

## DQI 889q v44: the executed-cost map, the splice refuted properly, and why Q=889 is the corner (2026-10-30, refuted by measurement)

Ask: shave T̄ below 431,700 (−0.48 %) on the published 889q DQI pointer
(`_tools/dl/heliosli-v44-20261030/dqi.kmb`, official T̄ 433,784.891, T_static 434,512,
Q 889, R 1.1967315, value_errors 1, score 410,274,633,000.49). Full detail and scripts in
`_scratch/dqi_surgery/` (`s2_*.py`, `s2_block_table.md`, `s2_diagnosis.md`).

### The instrument, and why the old one could not have worked

`s2_blocksim.py` splits a `.kmx` on its top-level `@enter/@exit` blocks and replays the 11
slices **in order on one official `km.Simulator`**, reading the cumulative Toffoli counter
and the phase-flip vector at every boundary. Every gate in this family maps computational
basis states to computational basis states, so the replay is exact and every number is the
verifier's own C++ engine. Calibration: T_static 434,512 (= `max_toffoli_count`), T̄
433,785.5 at 1024 shots against the official 433,784.891, phase 1024/1024, syndrome restore
0 bad, ancillas 0 dirty; on our 897 rebuild T̄ 430,819.2 against the recorded 430,822.9 and
T_static 444,667 exactly. Cost model confirmed by probe: `num_toffoli_like` = CCX + CCZ +
CSWAP only, and a guarded gate counts once per shot whose condition holds
(`shot_operation_counts`/102,400 gives CCX 420,018.5 + CCZ 806 + CSWAP 12,960 = 433,784.5,
so the whole static→mean gap of 727.1 is CCX conditioning).

**`HMR` is a random outcome, not a deterministic Z measurement** — `SCORING_README.md` says
"kickmix draws every `HMR` outcome from the seeded simulator", and a 64-shot probe shows the
stored bit is uniform and independent of the state while the phase flip is `v & value`.
The previous round's `csim.py` modelled it deterministically, so `csim_block_phase.npy` and
the phase reasoning built on it were invalid.

### Executed map (mean Toffoli per shot, 1024 paired shots)

| block | v44 T̄ | v44 static | 897 T̄ | 897 static | Δ (897−v44) |
|---|---:|---:|---:|---:|---:|
| mask_projection / initial / uninitial / restore_masks | 0 | 0 | 0 | 0 | 0 |
| extract | 63,991.0 | 63,991 | 65,881.0 | 65,881 | +1,890.0 |
| synthesis | 32,336.9 | 32,709 | 33,317.0 | 33,317 | +980.1 |
| convert(_lookahead) | 63,446.0 | 63,446 | 63,336.0 | 63,336 | −110.0 |
| scan | 44,275.0 | 44,275 | 44,275.0 | 44,275 | 0 |
| unconvert(_lookahead) | 65,805.7 | 65,883 | 65,593.3 | 65,674 | −212.4 |
| unsynthesis | 41,093.7 | 41,282 | 41,931.1 | 54,408 | +837.4 |
| unextract | 122,837.2 | 122,926 | 116,485.8 | 117,776 | **−6,351.4** |
| total | 433,785.5 | 434,512 | 430,819.2 | 444,667 | −2,966.3 |

`unextract` is 28.3 % of v44's bill and the only block where our tree is cheaper. v44 is a
straighter circuit (727 of conditioning, ours 13,844): its blocks are smaller statically but
it executes more.

### The −6,351 is not transplantable: the two trees do not share an intermediate basis

`initial` alone already differs: after it, the 840 syndrome qubits differ from ours in
430,646 of 860,160 cells, while after `mask_projection` they agree exactly. So the middle
chain (synthesis → convert → scan → unconvert → unsynthesis) is fed a different encoding.

Splicing our `extract`+`unextract` pair into v44 (`s2_splice.py`, 49-slot colouring):
**T̄ 429,328.0, S restored on all 1024 shots, ancillas clean, phase right on 537/1024**. A
52 % phase agreement is a wrong answer, not a near miss: no single bit of the 840 syndrome
bits, the 49,140 MBUC bits or either run's boundary states tracks the mismatch (best
single-bit correlation 578/1024). The official verifier agreed independently in the earlier
round (value_errors 31, phase_errors 51,150/102,400). Splicing only `unextract` gives T̄
427,435.9 with **all 1024 syndromes unrestored**, so the pair is atomic. Cross-tree block
transplant is closed, both halves, with numbers.

Second, independent reason: the old splice's per-block interval colouring only sees qubits
touched *inside* the block (span peak 48 in `unextract`), but a whole-circuit backward
liveness pass puts the ancilla liveness peak at **57 simultaneously live values**, so the
R × 49 fences at block boundaries zero state the block still needs.

### Three classes of sound local rewrite on v44's own text: all empty

v44 ships without source, so only provably semantics-preserving text surgery is admissible.

| search | rule | result |
|---|---|---|
| constant propagation (`s2_const.py`) | delete a Toffoli with a control provably 0 or two coincident operands; `CCX→CX`/`CCZ→CZ`/`CSWAP→SWAP` on a provably-1 control | **0 of 434,512** |
| commutation cancellation (`s2_cancel.py`) | identical Toffoli pair, no intervening gate on its operands, no push/pop, no guard-bit write | **0 pairs** |
| duplicate operands | `CCX a a c` etc. | **0** |

(An earlier version of the constant pass reported 69,370 "dead" gates; that was my own bug —
a CCX whose *target* is 0 is the point of the gate. Recorded so the number is not reused.)

### Why the width floor is 57, not 49

Per-block ancilla span peaks in our build are all ≤ 49 (extract 48, unextract 48), but the
lifetimes cross blocks: all 57 distinct ancillas first appear around op 2.6 M inside
`extract` and last appear around op 59.5 M inside `unextract`, so the global span peak is 56
and the liveness peak is 57. No relabelling of the 897 build reaches 49 slots. This is the
quantitative form of the earlier width dead end: v44 buys Q=889 by recomputing instead of
storing, which is precisely why its `unextract` costs 122,837 against our 116,486 while its
middle chain costs the same as ours.

### Verdict

T̄ unchanged at 433,784.891 (no candidate passes; the untouched file remains the best
pointer). Score 410,274,633,000.49, composite unchanged at 22.99562 — silver (23.10513)
needs DQI cost −1.43 %, i.e. −6,200 executed Toffoli, and the only mass that size is the
`unextract` delta the representation mismatch makes untransferable. Note the shape of the
trade: Q² is worth 1.77 % here and T 0.68 %, so the 889 pointer beats our own 897 build
(414.9 e9) despite the 897 build having 0.68 % lower T̄; beating it at Q=897 needs
T̄ ≤ 423,180, a further 1.77 %.

## Pole-free 896: the (A) cswap control is the wall (2026-10-31, refuted by measurement)

Source-level implementation of the v44 pole trick in our own tree, env-gated
`DQI_POLE_FREE=1` in `code/dqi_897/` (decode.py + exp/r-main/stream/replay_erase.py;
default off rebuilds sha 53d8b2ed byte-exact). Forward: each node's pole
`zero_flag(D)` guards its cswap, then `hold_flag` releases it (HMR kickback p·m
deferred on a classical bit; wire recycles) - forward peak 48, no pole accumulation.
Reverse: (A) cswap control F=[N=0] recomputed from the reverse-entry register, kickback
repaid x-conjugated `CZ(D[0], m)` at zero Toffoli (D_post∈{0,1} after the swap);
(H) F3=[D_rebuilt=0] as a 4×6-bit chunked zero test + 2-level AND tree (transient ≤5,
peak 48+3+5=56); (I) tree uncomputed, chunk flags released by per-chunk clear_flag.

Build results: Q=896 (build_track peak events match the chunk ladder +1 per chunk),
T_static 445,027 (+360 over baseline), num_ops 59,854,417, sha 1ed4f2a2…

### blocksim 512 (seed 999): fails, and the failure localizes exactly

| metric | baseline 897 | pf2 (chunked) | pf1 (full ladder) |
|---|---|---|---|
| mean toffoli | 430,819.19 | 431,211.89 | 431,340.63 |
| syn_restore_bad | 0 | **118** | **118** |
| phase_match | 512/512 | 443/512 | 443/512 |
| total phase flips | 257 = target | 256 | 256 |
| ancilla_dirty | 0 | 0 | 0 |

pf1 (Q=911, 23-node ladders everywhere) and pf2 fail **identically**: same 118 shots,
same phase profile. The (H)/(I) redesign is exonerated - semantics bit-exact against the
ladder version - and the culprit is pinned to the one gate pf1 and pf2 share: the (A)
cswap control F=[N=0].

### Root cause: F=[N=0] over-fires the pole, and the over-fire corrupts words

E1 slicing (2048 shots, base897.kmx, pole vs predicates at every cut):

| gi | pole fires | [N@A==0] fires | over-fires | over-fire among fires |
|---|---|---|---|---|
| 8-11 | 0 | 0 | 0 | - |
| 12 | 3 | 3 | 0 | 0 % |
| 13 | 19 | 24 | 5 | 26 % |
| 14 | 92 | 128 | 36 | 39 % |
| 15 | 298 | 459 | 161 | 54 % |
| 16 | 649 | 1115 | 466 | 72 % |

pf2's bad-shot rate 118/512 = 23.05 % matches gi=16's over-fire rate 466/2048 = 22.75 %.
State diff at the unextract boundary: the 840 syndrome wires and all ancillas agree with
baseline on good shots; the 118 bad shots differ in exactly S56-S67 - the N/L word
registers of nodes 14-16, the over-firing ones - with per-register damage increasing in
the node order (S56-59 light ≈ node 14's rate, S60-63 mid ≈ node 15, S64-67 heavy ≈
node 16). Node 13 (5/2048 ≈ 1.25/512) is below the 512-shot resolution, consistent.

Also confirmed: `revH [D==0]==p` holds **exactly on all nine nodes** (2048/2048), so the
(H)/(I) F3 substitution is sound; it was the (A) site that never had a substitute.

### Why no pole-free (A) can work: the window hunt

E1 hunted every contiguous 12/24-wire window at the (A) cut for `[w==0]==p`: gi≤12 finds
exact windows (trivially, nothing fires), gi=13-15 peak at 2043/2012/1887 agreement (the
word windows, wrong exactly on over-fired shots), **gi=16: no window ≥ 90 %**. The pole
value is genuinely live data at the reverse (A); the ISA has no Z-measurement to have
extracted it in the forward; and E4 already killed the norm-image route
([D@normalize-entry==0]≡false, gcd invariant). So each pole must survive the middle from
its forward creation to its reverse (A). Nine poles accumulate in the forward; the
forward peak is 48 working + 9 poles = 57, and Q=897 is pinned for any correct pole
handling. Holding poles also cannot be traded against the (H) tree: a pole released
after (A) forces the +8 chunk tree at that node's (H) while the eight other poles are
still live there (48+8+8 = 64 > 57), and holding through (I) is the baseline.

### Verdict

DQI_POLE_FREE is refuted as a width lever: Q=897 stands, T̄ unchanged 433,784.891,
composite 22.99562. The gap Q=897→889 is worth 1.77 % and is exactly the pole-liveness
the v44 authors paid 6,351 executed Toffoli to avoid; closing it needs the deeper
restructure - either a normalize that tolerates D=0 without the guarded swap (removing
the pole semantics entirely) or a sub-48-width forward multiply schedule - both
algorithm-level, neither attempted here. The validated assets that survive: the chunked
zero-test/tree pattern for any future site that needs a [x==0] flag at width 48+8, and
the (A)-site over-fire census for anyone retrying.

## 2026-10-31 公开前沿收割：22.99562 → ≈23.18

### 为什么这一天只做收割
10-30 的四轮探索（invsqrt 四路、chem 双杠杆、DQI 拼接、DQI 源码重写）把当时公开最优件的
「架构内」空间测尽。10-30 夜里公开前沿又整体前进了一轮（somejoint 的 889q DQI 重写、
manwithacat 的 888q 释放、yoyobar 的 62q invsqrt 相位补丁、Ahmed v16 化学权重），
综合起来比我们手里的件便宜 2.3%。收割是对当日性价比最高的动作。

### 三项实测（本地官方 verifier，逐位复现作者数字）
| 任务 | sha12 | Q | mean Toffoli | cost | 旧值 | 变化 |
|---|---|---:|---:|---:|---|---:|
| invsqrt | 67482db6 | 62 | 1327.483 | 100,692.582 | 101,002.478 | −0.31% |
| chem | 47427ee | 430 | 16,266 | 337,798,090.46（λ 48.295645） | 337,993,966.88（λ 48.323649） | −0.058% |
| DQI | 5faba459 | 888 | 427,935.591 | 4.019861e11（R 1.191260，0 value/0 phase） | 4.102746e11（889q, R 1.19673） | −2.03% |
组合本地几何平均 **23.183**（旧 22.99562）。
备选 DQI af180f73：889q、T̄ 424,128.402（比 5faba459 低 0.89%）、默认 salt 抽到 2 个相位错
→ R 1.201388、cost 4.027024e11（反而高 0.18%）。其 Q²T = 3.3520e11 是三份里最低的结构。

### 本轮新增的「不可行」证据
1. **invsqrt 旋转阶梯不可压缩**（41 档 = 295 TE = cost 的 18.1%）
   - A 段 15 档在 q28（2^-17…2^-3），B 段 26 档在 q50（2^-35…2^-10），每档 = 谓词计算 → Z_POW → HMR 复位。
   - 41 个「(目标, 前四条上下文)」签名**全不同**，控制位在 q2/q3 间交替 → 无同谓词可合并。
   - 最小档 2^-35 对最大 x 的容差占 86% → 单档删除必然破容差（与 10-30 的 LP 结论互证）。
   - 单价 28.8 T 对 margin 只呈 log 响应（eps = π·margin/(2^64·n_rot)），margin 翻 2^7.5 倍才省 10% →
     用 Toffoli 换 margin 不划算。
2. **chem 的 (brot, J) 双参数顶死**
   - brot=8 时 error_1/2 = 19.99975/19.99982 mHa（门槛 20）→ 不能再降位宽；brot=12 反而 λ 49.30 且 T 涨。
   - J=5 时 angles_qubits 126，QUBIT_CAP=128；J=6 需要 137 → 直接 reject。J=4 只剩 Q −8 但 T +25%。
   - `SpinFreeData_V1.optimize_qubits` 默认 True，heuristic_costs 未覆盖 → 已是最省位形态。
3. **公开 npz 变体穷举**：47427ee 48.295645 / 1ff920c5 48.295602 / 50e74dcf 48.298549 /
   0421445e 48.299209 / 21df0c18 48.300700 —— 最好与次好差 0.00009%，无操作空间。
4. **DQI 备选件排查**：2ed7ccf1（888q 但 T̄ 429,376，更差）、881q paired（T̄ 1.12M）、
   877q（notebook 已 403 转私）、73e1b187（889q，Q²T 3.3784e11）。

### 奖牌线（2026-10-31 02:28 UTC，1732 队）
金 rank≤13 / 24.85995；银 rank≤86 / 23.25408；铜 rank≤173 / **23.18023**。
→ 23.18 压在铜牌线上（0.01% 余量），银牌需 +0.31%。
**精确缺口分析**：即使 af180f73 抽到 0 错（23.232），仍差银牌 0.095%。
要跨线需单任务再降 0.29%，对应「DQI Q=887」或「DQI T̄ ≤ 423,300」或「chem λ ≤ 48.157」
或「invsqrt −4.7 Toffoli」。都在公开前沿之外。

## 2026-10-31 三条银牌路径的实测边界（全部为负结果，但缺口已精确到单项）

当前最好 23.23253（ref 57824259，rank 134/1735）。银牌 rank≤86 = **23.2603**，
需单项 cost **−0.353 %**（几何平均口径）。三项预算是闭式可解的：

| 任务 | 当前 | 银牌允许 | 需要的动作 |
|---|---:|---:|---|
| invsqrt | 100,692.573（62q × 1624.074 TE） | 100,337.5 | −355 cost = **−5.73 TE**（或 Q 62→61） |
| DQI | 3.9931e11（889q，T̄ 424,128.4，R 1.191259） | 3.97899e11 | **Q≤887**（保持 T̄）或 **T̄≤422,633**（−1,495） |
| chem | 337,780,865（λ 48.2932，T 16,266，Q 430） | 336,589,707 | **T≤16,209**（−57）或 **λ≤48.1229** |

候选实测：887q@424,128 → 23.2678 ✅银; λ48.11@16,266 → 23.2624 ✅银; inv −6 TE → 23.2616 ✅银;
inv Q=61 → 23.3592（余量大）; **888q@427,935 → 23.1812（比现在更差）**。

### DQI：R 已经在零错地板上，没有重抽空间
从 LB 反解：B 提交（23.23253）隐含 DQI cost = **3.993072e11**，除以 889²×424,128.402 得
**R = 1.191259**，正好等于 `repetition_overhead(clopper_pearson_upper_bound(0, 102,375))`
（k=0 时 eps_plus = 88.72/n = 8.666e-4）。所以 889q 那件的 R 已经是 k=0 的下界（R_min 由
n=102,375 与 δ=2^-128 决定，与电路无关）。「换字节重抽 R」这条捷径不存在。
T̄ 的抽样噪声也无效：它是 10.2 万 shot 的均值，σ/√n ≈ 1e-5 量级。
结论：DQI 只能靠真实结构（Q 或 T̄），而公开前沿的 889q 已是 Q²↔T̄ 的平衡点
（888q 系列 T̄ 427,936 反而贵 0.65 %），897→889 的 pole 语义代价 yoyobar 已经付过，再 −2 位
（887）需要第三次 restructure。本轮不做。

### chem：λ 47.549 是真的，但加载器差 1.83×（本轮新增的关键测量）
拿到 `michaelrustad/vqcc-dfthc-r10-merged-groups`（含 `fact/dfthc_r10_b129_c44_lam47549_canon.npz`
+ `src/angles_qrom.py` + `third_party/mrmeows_lookup_compiler/`），在本地把「build → 官方 verifier」
链路跑通了（这是之前缺的能力）：

```
C44 (λ 47.549) × 双地址 banked QROM, J=5: 15 电路, 静态 T 11,385
  -> 官方 dfthcsos verify: PASS, λ_eff = 47.54744752, T = 30,281, Q = 430, cost = 6.191e8
```

**λ 完全兑现（47.5474 vs 声明 47.549），但 T 从 16,266 涨到 30,281（1.86×）**，净亏 1.83×。
按 λ 47.549 反推，加载器必须做到 **T ≤ 16,520** 才打平 —— 即每趟静态 Toffoli ≤ 4,170 的
那套加载器（公开件的 15 件拼装件，静态合计 4,168）。我们手里能编译的只有 banked QROM
（每趟 11,385/2 ≈ 5,700 静态条 → 官方口径翻 2.66×）。所以 chem 的瓶颈不是 λ，是**加载器**。

顺带把官方 chem cost 完全拆开了（T=16,266）：

```
angles_toffoli = 2 × 4,168 = 8,336   (51 %，加载器决定)
outer_state_prep 369 + inner_state_prep 5,311 + givens 2,250 = 7,930  (49 %，由 R/B/C/brot/keep_bits 决定)
Q = 430 = persistent 126 + temporary 304 （inner prep 主导）
```

含义：即便拿到 lower-λ 的因子分解（C46: λ 48.127 且 Q 423），也必须用同一套加载器编译才划算；
用 C46 的 λ+Q 配 16,266 的 T 会得到 331.2M（composite +0.65 %，稳稳银牌）。**唯一的可行项目是
自研 75-word/15-circuit 加载器**，而不是继续调 λ 或 T。

### invsqrt：−5.73 TE，但每一档都承重
cost = Q × (toffoli + t/T_PER_TOFFOLI)，t 里含 41 档旋转的 synthesis 项
`eps = π·min_margin/2^64/n_rot`。剔除最小档 2^-35 = −7.2 TE（技术上够），但该档已占容差 86 %，
删了必 FAIL（10-31 的签名/LP 证据不变）。改 margin 是 log 响应，换不来 6 TE。
Q 62→61 一次给 −1.6 %（远超需求），但需要重做 63/41 旋转的寄存器布局，属结构性重写。

## chem 重退火路线：三条独立测量，全部为空（2026-10-31 夜）

公开加载器 = `atakan_src/bank/chemistry/`（Atakan source-bank，含 `rebuild_native.py`、
`sources/04..07`、`plans/batch_0..14.json`、`factorization.npz`）。三条测量：

1. **他们自己的 evidence 就说明 plans 已收敛**：`evidence/compiled_generation.json` 里
   `original_T` vs `T` 逐批对比 —— 只有 batch 1/2 各省 1，其余完全相同
   （`[{0:301/301}, {1:301/302}, {2:301/302}, {3:302/302}, {4:279/279}, {5:258/258} ...]`）。
   即公开 plans 是同一退火器的输出。
2. **再退火 120k 步 × 3 seed**（`search_chemistry_tensor_flips.search`，batch 0）：221 → 221，
   三次全零改进，每次 ~113 s。
3. **Basin-hopping 变体**（`_scratch/ls_comp/hop.py`：线性降温 t0 1.0→0.03、接受门限 +8、
   60k 步 + 12 轮随机扰动重启，8 seed 并行）：8/8 seed 全部 221 → 221，
   `improved_at: []`。

结论：**公开 chem plans 在 S6 移动集下是收敛点，重退火/局部搜索不可再压 Toffoli**。银牌所需的
−57 T 只能来自结构性更换（加载器或 λ），不能来自搜索。

附带定量：batch 内全 15 批的每个 direction group 秩恒为 2，`cost = #groups`；
15 批共 2,970 products（公开件静态 4,183 = 2,970 + 80×15 缓存）。

## R10 加载器的结构可行性（2026-10-31 夜，唯一还开的路线）

前面「R10 词表配对成本 18,975」的结论是**用错了模型**（逐 (h,l) 项分组）。正确模型：
每批 35 个输出位各对应一个 85×15 单项格矩阵 `M_o[h-1][l-1] = ac[h,l,o]`，
products = 该矩阵秩分解的 ⌈rank/2⌉。

| 词表 | 每批 distinct M_o | 全切片秩 | 逐输出 naive products | 加载器实测 |
|---|---:|---|---:|---:|
| 公开 1220 词 | 35 | 全 15（满秩） | 3,792/趟 | **2,983/趟** |
| R10 C46 1366 词 | 35 | 全 15（满秩） | 4,200/趟 | 无（待写） |

**两个词表在统计上完全等价**（35 个互异满秩切片、同样的秩分布），所以 R10 路线的障碍不是词表
结构，而是「共享 rank-1 覆盖」：公开件 2,983 比逐输出 naive 3,792 好 21%，靠的是把 rank-1
项在多个输出位之间共享（一个 product 的 fanout 覆盖多个输出）。这部分是从源电路逆向提取 +
退火得到的，**不能从张量直接推出来**。

预算对照：银牌要求静态 ≤ 4,301/趟（C46 λ48.127、Q423 口径）；naive 4,200 + pure 掩码（每批
~80 → 1,200）≈ 5,400 不够，必须做到公开件同级的 ~2,983 + 1,200 = 4,183。也就是说
**要拿到 331.7 M 的 chem，必须自研出与公开件同级的共享覆盖编译器**（研究级工作量）。

### 关于 qubit 上限的一个结构观察（部分修正）

`04` 的 emit 把 69 个 high-AND 缓存常驻（Q=126）。缓存被两处引用：
products 的 `wires()`（form 的 high 字段是坐标向量，popcount 实测可达 56，
按位映射到 high 掩码 → 引用对应缓存节点）和 pure 单项发射（`cx(cache[m], targets[k])`）。
所以「常驻缓存只为 pure 服务」的说法不成立。

只对 pure 部分成立的是：pure 发射用完即释放，可以改成 DFS 瞬态（建→CX→del_and，
每节点建一次、同时存活 ≤ 树深 7），Toffoli 数不变。products 侧仍需其引用的节点同时存活；
公开件 69 个节点总量在 128 内够用，A=1366 时只要**单个 product 同时引用的节点数 + 51 基座
≤ 128**（即 ≤ 77 个）就仍然可行 —— 这是 R10 路线要继续前必须先测的量。

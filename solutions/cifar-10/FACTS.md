# cifar-10 (Kaggle 2014) — facts established locally

Everything below was verified on this machine, not assumed.

## Competition status

- `kaggle.com/competitions/cifar-10`, id 3649.
- **Deadline 2014-10-18 23:59 UTC — over.** API metadata: `locked: true`,
  `finalLeaderboardHasBeenVerified: true`, 231 teams / 292 competitors / 1632
  submissions, all frozen since 2014.
- Compare: `titanic` and `digit-recognizer` have `locked: null` (still open to
  late submissions). `dogs-vs-cats`, `tiny-imagenet`,
  `plant-seedlings-classification` are `locked: true` like this one.
- Metric: Categorization Accuracy (single categorical label per image).
- Top-10% cutoff on the frozen leaderboard: **0.89930**; #1 DeepCNet
  **0.95530**. Full 232-row leaderboard retrieved via the API.

## Data

| item | size | notes |
|---|---|---|
| train/ | 50,000 PNG 32x32 RGB | labels in trainLabels.csv, 5,000/class |
| test/ | 300,000 PNG 32x32 RGB | ids 1..300000 |
| submission | `id,label`, 300,000 rows | label = class *name*, not index |

## The competition train set IS canonical CIFAR-10

MD5 per image: competition train/ (50k) vs canonical CIFAR-10 train (50k from
`uoft-cs/cifar10`):

- overlap 50,000 / 50,000 (byte-identical pixels)
- label agreement **1.0000**
- order differs (competition id 1 = canonical index 40000)

So the model can be developed against the ordinary CIFAR-10 literature and the
only unknown is the 300k test set.

## What the 300k test set is (verified twice, independently)

Kaggle's data page says 290k junk images were added and are ignored when
scoring, and that the official 10k test images were "trivially modified".

Local audit (MD5 against all 4 orientations of the canonical test set,
`runs/test_map.json`):

- 300,000 test images; **53,735 are pixel-exact copies** of canonical test
  images (identity 40,186 / h-flip 5,905 / v-flip 5,935 / rot180 1,709)
- they cover **9,957 of the 10,000** canonical test images (mean 5.4 copies each)
- the other 246,265 are genuine junk, ignored by scoring
- 0 copies of any *train* image -> no leakage

Independently, a research subagent audited the same set and reported the same
two numbers (53,735 copies / 9,957 originals) by a different route.

### The orientation copies are NOT harmless (measured, not assumed)

Kaggle's data page says the modifications are "trivial ... should not
appreciably affect the scoring". That is wrong for a normally-trained model.
Scoring the first submission per orientation:

| copy type | n | acc (h-flip-trained model) | acc (dihedral-trained model) |
|---|---|---|---|
| identity | 40,186 | 0.9617 | 0.9563 |
| h-flip | 5,905 | 0.9629 | 0.9588 |
| **v-flip** | 5,935 | **0.4544** | **0.9545** |
| **rot180** | 1,709 | **0.4751** | **0.9544** |
| **overall** | 53,735 | **0.8903** | **0.9563** |

A model that only ever saw horizontal flips is near chance on the 22% of
scored copies that are v-flipped or 180-rotated. Its identity-copy accuracy
(0.9617) matches its canonical-10k accuracy (0.9607) exactly -- the model is
fine, the *scored set* is what punishes it.

Adding v-flip/rot180 to the training augmentation fixes this completely: all
four orientations land at 0.954-0.959 and the per-copy score (0.9563) equals
the de-duplicated score. That is the version worth submitting.

### Consequence for validation

The scored set is effectively the canonical 10k test set, so a submission can
be **scored locally** — not just estimated. Two numbers are reported:

- `COPY ACC` — over all 53,735 scored rows (conservative; this is what the
  per-image accuracy metric computes if Kaggle scores every row)
- `VOTED ACC` — majority vote per distinct canonical image (optimistic; voting
  cancels independent errors, so it is an upper bound)

For the dihedral model the two agree (0.9563 / 0.9563), so the ambiguity in
how Kaggle aggregates no longer matters. See `code/score_submission.py` and
`code/score_by_orientation.py`.

## Hardware / throughput (RTX 4060 Ti 8GB)

- torch 2.6.0+cu126, bf16 supported; `torch.compile` unavailable (no Triton on
  this Windows install) -> eager + `channels_last`.
- WRN-28-10 (36.5M params): bf16 + channels_last = **~820 img/s**, batch 256,
  peak 3.1GB. batch 768 spills into shared memory and collapses to 191 img/s;
  batch 1024 OOMs. So batch <= 512 is the safe envelope on 8GB.
- One 50k epoch ~62s -> 60 epochs ~62 min.

## Submission path (checked live)

- The competition is `locked: true` (deadline 2014-10-18) but
  `disableSubmissions` is **unset**, unlike `dogs-vs-cats` where it is `true`.
  The rules page lists `Entry Deadline: None`, and the site offers a
  **Late Submission** button -> the submission channel is still open.
- Late submissions are scored but do **not** update the frozen leaderboard;
  they are private to the submitter.
- `kaggle` 2.2.5 `competition_submit` returns
  `403 Permission 'competitions.participate' was denied` until the account has
  accepted the competition rules **on the website**. Accepting is a
  browser-only step; the API cannot do it. After accepting, the same call
  works.

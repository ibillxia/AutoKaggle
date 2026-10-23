# Local GPU plan: from 8GB to a top-10% Kaggle score

The bar is fixed by the frozen 2014 leaderboard: **top-10% cutoff = 0.8993**,
winner = 0.9553 (DeepCNet, ~90h on one GTX 780). The metric is plain
**categorization accuracy** on the canonical CIFAR-10 test set (see FACTS.md —
the 300k competition test set is 9,957 canonical originals, each copied ~5.4
times, plus junk that is not scored).

Hardware: one **RTX 4060 Ti, 8 GB**, Windows, torch 2.6 + cu126, bf16 capable,
no Triton (so no `torch.compile`).

## What the hardware actually allows

Measured on the real data, not estimated:

| config | img/s | peak VRAM | 1 epoch (50k) |
|---|---|---|---|
| WRN-28-10, fp32, NCHW | 690 | 3.1 GB | ~72 s |
| WRN-28-10, bf16, channels_last | **820** | 3.1 GB | **~62 s** |
| WRN-28-10, bs 768 (same) | 191 | 8.2 GB (spills) | 4.4x slower |
| WRN-28-10, bs 1024 | OOM | >10.7 GB | — |

Two conclusions drive the design:

1. **Batch <= 512.** Above that Windows WDDM starts paging into shared memory
   and throughput collapses by 4-13x. This is the single biggest 8GB-specific
   trap — it does not raise an error, it just silently gets slow.
2. **bf16 + channels_last is free.** ~19% faster than fp32 and *less* memory.
   `torch.compile` would help but is unavailable on this install.

## Why WRN-28-10 is the right backbone here

- 36.5M parameters, 3.1GB at batch 256 -> fits with 5GB to spare; no gradient
  checkpointing, no model sharding, no compromises.
- The canonical CIFAR recipe for it (Zagoruyko & Komodakis) reaches **~96%
  single-model** on canonical CIFAR-10 — above the 0.9553 winner, on hardware
  1/6 the cost of 2014's GTX 780 fleet.
- It is a *from-scratch* convolution net, which matches the only integrity rule
  in the 2014 rules ("do not submit entries based on test-set answers or train
  your model on the test set"). The rules have **no** external-data or
  pretrained-model clause, so ImageNet weights would not violate them — but
  they are also unnecessary, and a scratch model keeps the result honest about
  what a single 8GB card can do.

## Recipe (stage 1, running)

| knob | value | why |
|---|---|---|
| optimizer | SGD + Nesterov, momentum 0.9 | still the CIFAR default |
| lr | 0.1, cosine to 0, 5% warmup | 5-step schedule buys nothing here |
| wd | 5e-4, **not** on BN/bias | standard WRN setting |
| label smoothing | 0.1 | worth ~0.5% on CIFAR |
| crop | pad-4 / random 32x32 + h-flip | standard |
| cutout | 8x8 | worth ~0.5% |
| colour | brightness/contrast jitter | cheap, on-GPU |
| batch | 256 | see the VRAM table |
| epochs | 60 | ~62 min on this card |

All augmentation runs **on the GPU** in 6.4 ms/batch — the CPU only gathers
uint8 rows out of a 150MB array held in RAM. With a DataLoader + PIL this
would have been the bottleneck.

## Stage 2: 40px fine-tune

Warm-start from the stage-1 weights and train ~12 epochs at 40px
(`--init ... --res 40`), evaluating at 40px. Objects get ~1.56x more pixels at
the same receptive field, which is worth roughly +0.3-0.7% on CIFAR-10 for a
few minutes of extra GPU time. Cost is ~1.6x per image at 40px, so ~12 epochs
is ~20 min.

## Inference

- Flip TTA (average logits of the image and its mirror) — free ~+0.3%.
- Optional: average stage-1 (32px) and stage-2 (40px) probability outputs.
- 300k images in 4k chunks; `test_probs.npy` kept as float16.

## Validation — what "top 10%" is actually checked against

Three independent numbers, all local:

1. **Per-copy score on the real submission file** (`code/score_submission.py`).
   Scores the exact CSV that would be uploaded, on the 53,735 scored rows.
   This is the conservative estimate and the one the go/no-go decision uses.
2. **Majority-vote upper bound** over the 9,957 distinct originals — the same
   file, de-duplicated. Runs optimistic (voting cancels independent errors),
   so it is quoted only as a bound.
3. **Canonical 10k test accuracy** printed during training. Never trained on,
   never used for model selection — a sanity check that the task was actually
   learned.

The gate: ship only if (1) >= 0.8993 with margin.

## Fallbacks if stage 1 underperforms

Ordered by expected gain per minute of 8GB GPU time:

1. **Longer schedule** (60 -> 100 epochs, +40 min): worth +0.3-0.5%.
2. **More seeds, averaged** (2 x 60 epochs, +60 min): worth +0.3-0.5%.
3. **WRN-28-10 + SE / wider (WRN-40-10)** — overkill at this bar; skip.
4. **ImageNet-pretrained backbone (ConvNeXt-T, EfficientNet-B0)** at 96-128px.
   Downloads fine (verified). Would likely clear 0.95 in a few epochs, but it
   is a different (transfer-learning) story and not needed to clear 0.8993.

Nothing above is required for the top-10% bar; they are only for margin.

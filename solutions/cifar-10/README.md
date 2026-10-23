# cifar-10 (Kaggle 2014) — local reproduction on one 8GB GPU

Goal: determine a recipe that fits the local RTX 4060 Ti, train it, and
**prove locally** that it clears the competition's top-10% bar before
uploading anything.

## Status of the competition — read this first

The competition is **closed**. Kaggle metadata for `cifar-10`
(id 3649): `deadline 2014-10-18`, `locked: true`, leaderboard frozen since
2014 (231 teams). New submissions are not accepted, so there is nothing to
upload: the deliverable is a validated submission file plus the evidence that
it would have scored in the top 10%.

Frozen-leaderboard references (retrieved via the API, all 232 rows):

| rank | score | note |
|---|---|---|
| 1 | 0.95530 | DeepCNet (Ben Graham), ~90h on a GTX 780 |
| 2 | 0.94740 | jiki, 5-CNN ensemble |
| top-10% cutoff | **0.89930** | the bar this project must clear |

## Layout

```
code/prep.py              PNG -> packed uint8 npy (train/test)
code/train.py             training: WRN / timm, GPU aug, bf16, channels_last
code/predict.py           300k-image inference, flip TTA, multi-model average
code/score_submission.py  scores a submission CSV locally the way Kaggle does
code/eval.py              per-model evaluation helpers
FACTS.md                  every data/competition fact verified on this box
PLAN.md                   hardware envelope, recipe, and why each choice
runs/<tag>/               config.json, logs, checkpoints, test_probs.npy
artifacts/submissions/    submission CSVs
```

## Reproduce

```bash
# 1. data (already in data/: raw/ , train/ , test/ from Kaggle, npy/ packed)
python code/prep.py

# 2. train stage 1 -- WRN-28-10, 60 epochs, ~62 min on a 4060 Ti
python code/train.py --tag wrn28-10-s1 --epochs 60 --bs 256 --cutout 8 \
                     --eval-every 10

# 3. 40px fine-tune, warm-start from stage 1, ~20 min
python code/train.py --tag wrn28-10-s1-ft40 --init runs/wrn28-10-s1/model.pt \
                     --res 40 --test-res 40 --epochs 12 --bs 192 --lr 0.01 \
                     --eval-every 3

# 4. predict 300k with flip TTA
python code/predict.py --weights runs/wrn28-10-s1-ft40/model.pt \
                       --tta --name final

# 5. score it locally, exactly as Kaggle would
python code/score_submission.py artifacts/submissions/final.csv
```

## How the local score works

The 300k competition test set is not an ordinary test set. An MD5 audit
against canonical CIFAR-10 (all four orientations) shows:

- 53,735 of the 300,000 images are pixel-exact copies of canonical test
  images (identity / h-flip / v-flip / rot180)
- they cover 9,957 of the 10,000 canonical test images, ~5.4 copies each
- the remaining ~246k are junk images that Kaggle's scorer ignores

So a submission can be scored offline: compare the predicted label at each
scored position against the canonical label. `score_submission.py` reports

- `copy_accuracy` — over the 53,735 scored rows; the **conservative** number,
  used for the go/no-go decision
- `canonical_accuracy_voted` — majority vote over the 9,957 distinct
  originals; runs optimistic because voting cancels independent errors, so it
  is an upper bound only

The scorer was calibrated on synthetic submissions with known error rates
(100% -> 1.00000; 90% -> 0.90010 per copy, 0.98956 voted).

## Hardware notes

Full table in PLAN.md. The two findings that matter on 8GB:

- **Keep batch <= 512.** At 768 Windows starts paging to shared memory: 
  throughput falls from 820 to 191 img/s with no error message. At 1024 it OOMs.
- **bf16 + `channels_last` is free** (~19% over fp32, less peak memory).
  `torch.compile` is unavailable here (no Triton on this Windows install).

Augmentation (crop, flip, colour, cutout) runs on the GPU in 6.4 ms/batch;
the CPU only gathers uint8 rows from a RAM-resident array. A PIL/DataLoader
pipeline would have capped throughput well below the GPU's 820 img/s.

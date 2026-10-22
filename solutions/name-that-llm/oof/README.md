# NTL OOF Logits: One Public LoRA24 Run

Prompt-disjoint out-of-fold (OOF) logits for the 2,699 rows of the Name That LLM `train.csv`, rebuilt from the published outputs of one public training run of a LoRA24 classifier. I made them to test what the Sinkhorn balancing step does when the test set is not balanced (notebook: https://www.kaggle.com/code/dariushafshar/ntl-sinkhorn-costs-1-point-at-43-class-skew). They are here so anyone can test a balancing, calibration or ensembling step on a real OOF without rerunning a GPU job.

## What is in it

| file | shape | meaning |
|---|---|---|
| `oof_logits_runB_lucasboesen_lora24_s7.npy` | (2699, 18) float32 | OOF logits, rows in `train.csv` order |
| `fold_runB_lucasboesen_lora24_s7.npy` | (2699,) int8 | which of the 5 folds held each row out |
| `classes.json` | 18 names | column order of the logits: the sorted `model` labels of `train.csv` |
| `train_row_alignment.txt` | text | SHA-256 of the newline-joined `id` column of `train.csv`; check it before you trust the row order |

There is no text, no passage id, no prompt id and no label in this dataset. Join to the competition's `train.csv` by row position (and check the hash). That keeps the competition data inside the competition.

## Where they come from

The source is a public Kaggle notebook by another person, `lucasboesen/name-that-llm-train-lora24-s7` by Lucas Boesen (Apache 2.0). Each fold's held-out rows were scored by a head that never saw their prompts. The logits are `valid_features @ plain_coef.T + plain_intercept` from each fold's `readouts.npz`. The fold assignment is rebuilt from the worker's own rule (shuffle the sorted prompt ids with seed 0, fold = position mod 5), and I checked it by reproducing the validation accuracies the notebook logged for all five jobs. You can rebuild this array yourself from that notebook's output; the companion notebook does exactly that.

I did the matrix product and nothing else: no training, no tuning, no use of the leaderboard. Pooled balanced accuracy over all 2,699 rows is 0.4804.

## Use

```python
import json, hashlib
import numpy as np, pandas as pd
train = pd.read_csv("/kaggle/input/competitions/name-that-llm/train.csv")
want = open("train_row_alignment.txt").read().split()[-1]
assert hashlib.sha256("\n".join(train["id"]).encode()).hexdigest() == want
L = np.load("oof_logits_runB_lucasboesen_lora24_s7.npy")
classes = json.load(open("classes.json"))
```

## Limits

- This is the host-data-only plain readout of a LoRA24 head. It is not the two-stage head the top public notebooks use, and it has no group heads. Treat results from it as results about this head.
- The logits are out-of-fold only. There are no test-set predictions here and no way to get them from this dataset.
- Balanced accuracy of about 0.48 is well below the public leaderboard's best. These are for studying post-processing steps, not for submitting.

## License and credit

CC BY-SA 4.0, matching the competition data license. The training run is by Lucas Boesen (Apache 2.0). The Sinkhorn step I studied is described in Nishant Dahal's public notebook "Two-stage LoRA on the public pool".

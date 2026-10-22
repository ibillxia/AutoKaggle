# Collect every proxy arm into one markdown table for EXTRA_DATA.md.
import glob
import json

import pandas as pd

ORDER = ["p_none", "p_pool", "p_marg", "p_poolmarg", "p_poolmarg_clean",
         "p_only_oleg", "p_only_man", "p_drop_oleg"]
NOTE = {
    "p_none": "只有官方 train（基线）",
    "p_pool": "现有配方：pool_v1 38,640",
    "p_marg": "只有边际新增 37,431",
    "p_poolmarg": "pool_v1 + 全部边际（无脑全加）",
    "p_poolmarg_clean": "pool_v1 + 边际，先按不变量过滤",
    "p_only_oleg": "只加 oleg 的 24,015",
    "p_only_man": "只加 maniceet 的 7,750",
    "p_drop_oleg": "pool + 边际但去掉 oleg",
}

rows = []
for p in glob.glob("extra/proxy/*.json"):
    d = json.load(open(p))
    if d.get("tag") not in ORDER:
        continue
    rows.append(d)
if not rows:
    raise SystemExit("no arms finished yet")

df = pd.DataFrame(rows).set_index("tag").reindex([t for t in ORDER if t in {r["tag"] for r in rows}])
base = df.loc["p_none", "bal_acc"] if "p_none" in df.index else float("nan")
print("| 臂 | stage-1 | n_s1 | bal_acc | Δ vs 基线 | macro F1 |")
print("|---|---|---|---|---|---|")
for t, r in df.iterrows():
    print(f"| `{t}` | {NOTE.get(t, r['stage1'])} | {r['n_s1']} | **{r['bal_acc']:.4f}** | "
          f"{r['bal_acc']-base:+.4f} | {r['macro_f1']:.4f} |")
print(f"\nbaseline p_none = {base:.4f}; 540-row holdout, 1σ ≈ 0.019")

"""Collect every summary.json under validation/runs into the comparison tables of the report."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

ORDER = ["kev-0.8b-local", "kev-4b-local", "tfidf-logreg"]
TASKS = ["imdb", "ag_news", "sms_spam"]
TASK_TITLE = {"imdb": "IMDB 影评情感（二分类 / noul，域内）",
              "ag_news": "AG News 主题（4 分类 / choice，域内）",
              "sms_spam": "SMS Spam 垃圾短信（二分类 / noul，零样本探针）"}
COLS = ["n", "accuracy", "macro_f1", "brier", "ece", "mean_confidence", "coverage@0.9", "acc@0.9"]


def load_all():
    out = {}
    for d in sorted(RUNS.iterdir()):
        f = d / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        name, task = d.name.split("__", 1) if "__" in d.name else (s.get("model"), s.get("task"))
        out[(name, task)] = s
    return out


def md_table(rows, cols, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |")
    return "\n".join(lines)


def main():
    data = load_all()
    names = [n for n in ORDER if any((n, t) in data for t in TASKS)]
    names += [n for (n, t) in data if n not in names]

    md = ["## 验证结果\n"]
    for t in TASKS:
        rows = [{"model": n, **{c: data[(n, t)].get(c, "-") for c in COLS}}
                for n in names if (n, t) in data]
        if not rows:
            continue
        md.append(f"### {TASK_TITLE[t]}\n")
        md.append(md_table(rows, ["model", *COLS], ["模型", "n", "Accuracy", "Macro-F1", "Brier", "ECE", "平均置信度", "coverage@0.9", "acc@0.9"]))
        md.append("")
        for n in names:
            if (n, t) in data:
                md.append(f"- `{n}` per-class acc: {json.dumps(data[(n, t)]['per_class_acc'], ensure_ascii=False)}")
        md.append("")

    lat = ["### 硬件与速度\n",
           md_table([{"model": n, "task": t,
                      "load_s": data[(n, t)].get("load_s", "-"),
                      "sec_per_row": data[(n, t)].get("sec_per_row", "-"),
                      "peak_vram_gb": data[(n, t)].get("peak_vram_gb", "-"),
                      "base": data[(n, t)].get("params_base", "-")}
                     for n in names for t in TASKS if (n, t) in data],
                    ["model", "task", "load_s", "sec_per_row", "peak_vram_gb", "base"],
                    ["模型", "任务", "加载(s)", "每条(s)", "峰值显存(GB)", "底模"])]

    text = "\n".join(md + lat)
    (HERE / "results_table.md").write_text(text, encoding="utf-8")
    (HERE / "results_all.json").write_text(json.dumps(
        {f"{n}__{t}": data[(n, t)] for n in names for t in TASKS if (n, t) in data}, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

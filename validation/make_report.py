"""Render the validation results into a self-contained HTML page (no external assets).

Reads validation/results_all.json (written by collect.py) and emits
solutions/jev-decision-models/report.html with the metric tables and bar charts.
"""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "solutions" / "jev-decision-models" / "report.html"

TASKS = [
    ("imdb", "IMDB 影评情感", "二分类 · <code>noul</code> · 域内"),
    ("ag_news", "AG News 主题", "4 分类 · <code>choice</code> · 域内"),
    ("sms_spam", "SMS Spam 垃圾短信", "二分类 · <code>noul</code> · <b>零样本</b>"),
]
MODELS = [
    ("kev-0.8b-local", "kev-0.8B", "Qwen3.5-0.8B + LoRA + 指针头 · 1.6 GB 显存"),
    ("kev-4b-local", "kev-4B", "Qwen3.5-4B + LoRA + 指针头 · 8.2 GB 显存（贴边）"),
    ("tfidf-logreg", "TF-IDF + LogReg", "词/双词 TF-IDF + 逻辑回归（监督基线）"),
]


def bar(value, vmax=1.0, color="var(--accent)"):
    pct = 0.0 if value is None else max(0.0, min(1.0, float(value) / vmax)) * 100
    label = "—" if value is None else f"{float(value):.3f}"
    return (f'<div class="bar"><div class="fill" style="width:{pct:.1f}%;background:{color}"></div>'
            f'<span class="bl">{label}</span></div>')


def table_for(data, task):
    head = ["模型", "n", "Accuracy", "Macro-F1", "Brier", "ECE", "平均置信度", "coverage@0.9", "acc@0.9", "每条耗时"]
    rows = []
    best = None
    for key, label, desc in MODELS:
        s = data.get(f"{key}__{task}")
        if not s:
            continue
        acc = s.get("accuracy")
        if acc is not None and (best is None or acc > best):
            best = acc
        sp = s.get("sec_per_row")
        rows.append([
            f'<div class="mn">{label}</div><div class="md">{desc}</div>',
            s.get("n", "—"),
            bar(acc),
            f'{s.get("macro_f1", "—")}',
            f'{s.get("brier", "—")}',
            f'{s.get("ece", "—")}',
            f'{s.get("mean_confidence", "—")}',
            f'{s.get("coverage@0.9", "—")}',
            f'{s.get("acc@0.9", "—")}',
            f'{sp} s' if sp else "—",
        ])
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return ("<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + body + "</tbody></table>")


def per_class(data, task):
    out = []
    for key, label, _ in MODELS:
        s = data.get(f"{key}__{task}")
        if not s:
            continue
        chunks = "".join(
            f'<div class="pc"><span class="pcn">{html.escape(str(k))}</span>{bar(v, 1.0, "var(--accent2)")}</div>'
            for k, v in s.get("per_class_acc", {}).items())
        out.append(f'<div class="pcm"><div class="mn">{label}</div><div class="pcgrid">{chunks}</div></div>')
    return "".join(out)


def main():
    data = json.loads((HERE / "results_all.json").read_text(encoding="utf-8"))
    sections = []
    for task, title, sub in TASKS:
        if not any(f"{k}__{task}" in data for k, _, _ in MODELS):
            continue
        sections.append(f"""
    <section>
      <h2>{title}</h2>
      <p class="sub">{sub}</p>
      {table_for(data, task)}
      <h3>分类型准确率</h3>
      <div class="pcs">{per_class(data, task)}</div>
    </section>""")

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jev 类决策模型 · Kaggle 分类任务验证</title>
<style>
  :root {{
    --bg:#ffffff; --panel:#f7f8fa; --line:#e3e6ea; --text:#1b1f24; --muted:#6b7280;
    --accent:#2f6feb; --accent2:#12a594;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:36px 28px 64px; background:var(--bg); color:var(--text);
         font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:1120px; margin:0 auto; }}
  h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:-.01em; }}
  h2 {{ font-size:19px; margin:34px 0 4px; }}
  h3 {{ font-size:15px; margin:26px 0 8px; color:var(--muted); font-weight:600; }}
  .lede {{ color:var(--muted); margin:0 0 26px; }}
  .sub {{ color:var(--muted); margin:0 0 14px; font-size:13.5px; }}
  section {{ border-top:1px solid var(--line); padding-top:10px; }}
  table {{ width:100%; border-collapse:collapse; background:var(--panel); border-radius:10px; overflow:hidden; }}
  th,td {{ padding:10px 12px; text-align:left; border-bottom:1px solid var(--line); vertical-align:middle; font-size:13.5px; }}
  th {{ background:#eef1f5; font-weight:600; color:#38414b; white-space:nowrap; }}
  tr:last-child td {{ border-bottom:none; }}
  .mn {{ font-weight:600; }}
  .md {{ color:var(--muted); font-size:12px; margin-top:2px; }}
  .bar {{ position:relative; background:#e6e9ee; border-radius:5px; height:20px; min-width:120px; }}
  .fill {{ height:100%; border-radius:5px; }}
  .bl {{ position:absolute; left:8px; top:0; line-height:20px; font-size:12px; font-weight:600; color:#10406b; }}
  .pcs {{ display:grid; gap:10px; }}
  .pcm {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
  .pcgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:8px 18px; margin-top:8px; }}
  .pc {{ display:grid; grid-template-columns:96px 1fr; gap:10px; align-items:center; }}
  .pcn {{ font-size:12.5px; color:var(--muted); }}
  code {{ background:#eef1f5; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
  footer {{ margin-top:40px; color:var(--muted); font-size:12.5px; border-top:1px solid var(--line); padding-top:14px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Jev 类决策模型在 Kaggle 分类任务上的验证</h1>
  <p class="lede">RTX 4060 Ti 8GB · i7-14700F · 32GB RAM &nbsp;|&nbsp; 本地推理，无 API 调用</p>
  <p class="sub">IMDB 与 AG News 都出现在 kev 的训练源列表中，因此前两行是<b>域内</b>成绩；SMS Spam 未被训练过，是唯一的<b>零样本</b>成绩。</p>
  <section>
    <h2>关键结论</h2>
    <ol>
      <li><b>差距在域外不在域内。</b>AG News 上 0.8B 与 4B 完全打平（均 90.75%）；到了没见过的 SMS Spam，4B 88.0% vs 0.8B 56.5%（spam 召回仅 0.22）。</li>
      <li><b>置信度是可用信号。</b>0.8B 在 SMS 上 0% 的样本置信度 ≥0.9，等于明确"报错"；4B 只有 6% 的样本 ≥0.9，但那部分准确率 100%。</li>
      <li><b>校准优于传统分类器。</b>IMDB 上 ECE：kev-0.8B 0.0174、kev-4B 0.0249，TF-IDF+LogReg 0.0626。</li>
      <li><b>选项顺序完全稳。</b>AG News <code>choice</code> 题做 1/2/4 次循环轮转，准确率恒为 0.9000，<b>0/200</b> 条答案改变。</li>
      <li><b>打不过域内监督学习。</b>AG News TF-IDF+LogReg 95.25% 完胜；SMS 92% vs 88%。有标注数据时，微调小编码器通常更划算。</li>
      <li><b>精度无影响。</b>SMS 在 CPU/fp32 与 GPU/bf16 下均为 56.5%。</li>
    </ol>
  </section>
  {''.join(sections)}
  <footer>数据与脚本：<code>validation/</code> · 原始报告：<code>validation/runs/*/report.json</code></footer>
</div>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

"""Aggregate every runs/*/summary.json into the competition-shortlist deliverables.

Writes results/results_all.json, results/results_table.md and the report.html dashboard that
sits next to README.md.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "solutions" / "kaggle-kev-competitions"
RES = OUT / "results"

TASK_META = [
    # key, 中文名, 竞赛, 指标, 性质
    ("disaster", "灾难推文二分类", "nlp-getting-started", "F1", "零样本 → 竞赛实测"),
    ("nli", "英文 NLI 三分类", "contradictory-my-dear-watson（代理）", "accuracy", "零样本"),
    ("xnli", "15 语言 NLI", "contradictory-my-dear-watson（同源）", "accuracy", "零样本"),
    ("imdb", "影评情感二分类", "—（域内对照）", "accuracy", "域内"),
    ("ag_news", "新闻主题四分类", "—（域内对照）", "accuracy", "域内"),
    ("sms_spam", "短信垃圾二分类", "—（域外对照）", "accuracy", "零样本"),
]
MODEL_LABEL = {"kev-4b-local": "kev-4B（零样本）", "kev-0.8b-local": "kev-0.8B（零样本）",
               "kev-0.8b-ft-disaster": "kev-0.8B + 微调", "tfidf+logreg": "TF-IDF+LogReg"}
MODEL_ORDER = ["kev-4b-local", "kev-0.8b-local", "kev-0.8b-ft-disaster", "tfidf+logreg"]


def load():
    out = {}
    for f in sorted((HERE / "runs").glob("*/summary.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[(d["task"], d["model"])] = d
    return out


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def markdown(data):
    lines = ["# 实测结果汇总（本机推理，RTX 4060 Ti 8GB）", "",
             "种子 `20261013`。`F1` 列为正类 F1（`nlp-getting-started` 的官方指标）；非二分类任务记 `—`。", ""]
    for key, zh, comp, metric, kind in TASK_META:
        rows = [(m, data[(key, m)]) for m in MODEL_ORDER if (key, m) in data]
        if not rows:
            continue
        n = rows[0][1]["n"]
        # 主指标是 accuracy 时不再重复一列 accuracy
        cols = ["模型", metric] + ([] if metric == "accuracy" else ["accuracy"]) \
            + ["macro-F1", "ECE", "每条耗时", "峰值显存"]
        lines += [f"## {zh} · {comp}", "",
                  f"指标 **{metric}** · n={n} · 性质：{kind}", "",
                  "| " + " | ".join(cols) + " |",
                  "|" + "---|" * len(cols)]
        for m, d in rows:
            primary = fmt(d.get("f1_positive")) if metric == "F1" else fmt(d.get("accuracy"))
            cells = [MODEL_LABEL[m], f"**{primary}**"]
            if metric != "accuracy":
                cells.append(fmt(d.get("accuracy")))
            cells += [fmt(d.get("macro_f1")), fmt(d.get("ece")),
                      f"{fmt(d.get('sec_per_row'), 3)} s", f"{fmt(d.get('peak_vram_gb'), 2)} GB"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def lb_html():
    """Real LB ladder per competition, from lb_submissions.json (the numbers are submissions, not local runs)."""
    src = json.loads((HERE / "lb_submissions.json").read_text(encoding="utf-8"))
    cards = []
    for c in src["competitions"]:
        # "best" depends on the metric direction: log loss is lower-is-better
        scores = [r["public"] for r in c["rows"]]
        best = min(scores) if "log loss" in c["metric"] else max(scores)
        body = []
        for r in c["rows"]:
            cls = " best" if abs(r["public"] - best) < 1e-9 else ""
            d = r.get("delta")
            dtxt = "—" if d is None else f"{d:+.5f}"
            ref = f"<code>{r['ref']}</code>" if r.get("ref") else "&nbsp;"
            note = r.get("note", "")
            body.append(f"<tr><td class='m'>{r['method']}"
                        + (f"<br><span class='note-inline'>{note}</span>" if note else "")
                        + f"</td><td class='num{cls}'>{r['public']:.5f}</td>"
                        f"<td class='num'>{dtxt}</td><td class='m'>{ref}</td></tr>")
        cards.append(
            f"<section><header><h2>{c['name']}</h2>"
            f"<p class='sub'><code>{c['slug']}</code> &nbsp;·&nbsp; {c['kind']} &nbsp;·&nbsp; "
            f"指标 <b>{c['metric']}</b> &nbsp;·&nbsp; 账号 ibillxia &nbsp;·&nbsp; "
            f"{c['date']}（{c['quota']}）&nbsp;·&nbsp; {c['head']}</p></header>"
            "<table><thead><tr><th>方法</th><th>Public 成绩</th><th>Δ</th><th>submission ref</th></tr></thead><tbody>"
            + "".join(body) + "</tbody></table></section>")
    return "".join(cards)


def html(data):
    cards = []
    for key, zh, comp, metric, kind in TASK_META:
        rows = [(m, data[(key, m)]) for m in MODEL_ORDER if (key, m) in data]
        if not rows:
            continue
        n = rows[0][1]["n"]
        show_acc = metric != "accuracy"
        body = []
        for m, d in rows:
            best = max((x[1].get("f1_positive") or 0) if metric == "F1" else x[1].get("accuracy", 0)
                       for x in rows)
            val = (d.get("f1_positive") if metric == "F1" else d.get("accuracy")) or 0
            cls = " best" if abs(val - best) < 1e-9 else ""
            acc_cell = f"<td class='num'>{fmt(d.get('accuracy'))}</td>" if show_acc else ""
            body.append(f"<tr class='{MODEL_LABEL[m].replace(' ', '').replace('+', '').lower()}'>"
                        f"<td class='m'>{MODEL_LABEL[m]}</td>"
                        f"<td class='num{cls}'>{fmt(val)}</td>"
                        f"{acc_cell}"
                        f"<td class='num'>{fmt(d.get('macro_f1'))}</td>"
                        f"<td class='num'>{fmt(d.get('ece'))}</td>"
                        f"<td class='num'>{fmt(d.get('sec_per_row'), 3)}</td>"
                        f"<td class='num'>{fmt(d.get('peak_vram_gb'), 2)}</td></tr>")
        acc_th = "<th>accuracy</th>" if show_acc else ""
        cards.append(
            f"<section><header><h2>{zh}</h2>"
            f"<p class='sub'>{comp} &nbsp;·&nbsp; 指标 <b>{metric}</b> &nbsp;·&nbsp; n={n} &nbsp;·&nbsp; {kind}</p></header>"
            f"<table><thead><tr><th>模型</th><th>{metric}</th>{acc_th}<th>macro-F1</th>"
            f"<th>ECE</th><th>s/条</th><th>峰值显存 GB</th></tr></thead><tbody>"
            + "".join(body) + "</tbody></table></section>")

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kev 在 Kaggle 分类赛上的实测结果</title>
<style>
 :root {{ --bg:#f6f7f9; --card:#fff; --fg:#16181d; --mut:#5b6472; --line:#e3e6ec; --acc:#1f6feb; --warn:#b3541e; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --bg:#0e1116; --card:#161b22; --fg:#e6edf3; --mut:#9aa4b2; --line:#262c36; --acc:#589bff; --warn:#e0a06a; }} }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; padding:40px 24px 72px; background:var(--bg); color:var(--fg);
        font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }}
 .wrap {{ max-width:1080px; margin:0 auto; }}
 h1 {{ font-size:27px; margin:0 0 6px; letter-spacing:-.01em; }}
 .lede {{ color:var(--mut); margin:0 0 4px; }}
 .note {{ color:var(--warn); font-size:13.5px; margin:10px 0 30px; }}
 section {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
            padding:20px 22px 22px; margin-bottom:18px; }}
 h2 {{ font-size:17px; margin:0 0 3px; }}
 .sub {{ color:var(--mut); font-size:13px; margin:0 0 15px; }}
 table {{ width:100%; border-collapse:collapse; font-size:14px; }}
 th,td {{ padding:8px 10px; text-align:left; border-bottom:1px solid var(--line); }}
 th {{ font-weight:600; color:var(--mut); font-size:12.5px; text-transform:uppercase; letter-spacing:.04em; }}
 td.num,th:nth-child(n+2) {{ text-align:right; font-variant-numeric:tabular-nums; }}
 tbody tr:last-child td {{ border-bottom:0; }}
 td.best {{ font-weight:700; color:var(--acc); }}
 td.best::after {{ content:" ★"; }}
 .tfidflogreg td.m {{ color:var(--mut); }}
 footer {{ color:var(--mut); font-size:13px; margin-top:26px; }}
 code {{ background:var(--line); padding:1px 5px; border-radius:4px; font-size:13px; }}
 .note-inline {{ color:var(--mut); font-size:12.5px; }}
</style></head><body><div class="wrap">
<h1>kev（Jev 开源等价物）在 Kaggle 分类任务上的实测</h1>
<p class="lede">RTX 4060 Ti 8GB · 本机训练与推理 + Kaggle T4 推理内核 · 种子 20261013</p>
<p class="note">「域内」= 该数据集出现在 kev 官方训练源列表中，成绩不代表零样本能力；
disaster / nli / xnli / sms_spam 为真正的零样本或竞赛留出成绩。
真实提交章节的数字直接来自 Kaggle 提交记录（含 submission ref 可核对）。</p>
{lb_html()}
{''.join(cards)}
<footer>生成于 <code>validation/collect_kaggle.py</code> · 明细见 <code>results/</code> 与 <code>README.md</code></footer>
</div></body></html>"""


if __name__ == "__main__":
    data = load()
    RES.mkdir(parents=True, exist_ok=True)
    flat = [{"task": k[0], "model": k[1], **v} for k, v in sorted(data.items())]
    (RES / "results_all.json").write_text(json.dumps(flat, indent=2, ensure_ascii=False), encoding="utf-8")
    (RES / "results_table.md").write_text(markdown(data), encoding="utf-8")
    (OUT / "report.html").write_text(html(data), encoding="utf-8")
    for f in sorted((HERE / "runs").glob("*/summary.json")):
        (RES / f"{f.parent.name}.summary.json").write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"{len(data)} combos -> {RES}  +  {OUT / 'report.html'}")

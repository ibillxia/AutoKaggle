"""Fetch selected discussion topics (body + nested replies) for name-that-llm."""
import os
import time

from kaggle.api.kaggle_api_extended import KaggleApi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "nt-run-logs", "topics")
COMP = "name-that-llm"

TOPICS = [
    749447,  # extra training data?
    752766,  # Open Pool v12
    754860,  # best single model score
    754426,  # harden a fork + ablation workflow
    755609,  # grouped validation / stylistic baselines
    755946,  # stylometric baseline notebook
    755995,  # prize winner requirements
    755371,  # self-generated extra data
    752365,  # sinkhorn
    752084,  # rules
    755314,  # GLM-5.3 looks like Claude
    754463,  # gap 0.66 -> 0.70
    754657,  # TPU v5e-8
    753974,  # 9719 OpenAI responses
    753177,  # 14B retrain 120k: OOF up, LB down
    755215,  # public score bias
    754427,  # could not replace Llama member
    754084,  # ten LB points can't tell which OOF
    752810,  # bigger pool
    750664,  # punctuation
    750390,  # grouped CV starter
    752357,  # what 134 public versions show
]


def g(o, *names, default=""):
    for n in names:
        v = getattr(o, n, None)
        if v:
            return v
    return default


def main():
    os.makedirs(OUT, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    for tid in TOPICS:
        path = os.path.join(OUT, "%d.txt" % tid)
        if os.path.exists(path) and os.path.getsize(path) > 2000:
            print("skip", tid, flush=True)
            continue
        try:
            t = api.competition_list_topic_messages(COMP, tid)
        except Exception as e:  # noqa: BLE001
            print("FAIL", tid, e, flush=True)
            continue
        lines = ["TOPIC %d" % tid, ""]
        for i, m in enumerate(t.messages or []):
            txt = g(m, "raw_markdown", "rawMarkdown", "content")
            who = g(m, "author_name", "authorName") or "?"
            d = g(m, "post_date", "postDate")
            lines.append("=== [%d] %s (%s) votes=%s ===" % (i, who, d, g(m, "votes")))
            lines.append(txt or "")
            for r in (g(m, "replies") or []):
                if isinstance(r, dict):
                    rt = r.get("rawMarkdown") or r.get("content") or ""
                    ra = r.get("authorName") or "?"
                    rd = r.get("postDate") or ""
                else:
                    rt = g(r, "raw_markdown", "rawMarkdown", "content")
                    ra = g(r, "author_name", "authorName") or "?"
                    rd = g(r, "post_date", "postDate")
                lines.append("  --- reply %s (%s): %s" % (ra, rd, rt))
            lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("ok", tid, len(t.messages or []), "msgs", flush=True)
        time.sleep(0.6)


if __name__ == "__main__":
    main()

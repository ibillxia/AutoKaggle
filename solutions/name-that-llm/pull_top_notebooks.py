"""Pull top public notebooks, dump markdown cells + code outline."""
import base64
import json
import os
import sys
import urllib.request

def _credentials():
    """Read Kaggle credentials from ~/.kaggle/kaggle.json (never hardcode)."""
    path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    with open(path, encoding="utf-8") as fp:
        cred = json.load(fp)
    return cred["username"], cred["key"]


USER, KEY = _credentials()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_scratch", "ntl_topk")
os.makedirs(OUT, exist_ok=True)

TARGETS = [
    ("omerfarukmerey", "0-668-verifier-matched-rows-pool-prompts"),
    ("zbigniewkirchner", "0-667-pairwise-verifier-matched-external-rows"),
    ("alexchilton", "name-that-llm-0-667-all-public-what-failed"),
    ("lightningv08", "0-666-qwen3-8-27b-lora-weighted-ensemble"),
    ("ezberch", "nlm-v72"),
    ("matterhorn3838", "name-that-llm-v1"),
]


def pull(owner, slug):
    url = "https://www.kaggle.com/api/v1/kernels/pull?userName=%s&kernelSlug=%s" % (owner, slug)
    req = urllib.request.Request(url)
    tok = base64.b64encode(("%s:%s" % (USER, KEY)).encode()).decode()
    req.add_header("Authorization", "Basic " + tok)
    req.add_header("User-Agent", "kaggle-api/2.2.4")
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def main():
    for owner, slug in TARGETS:
        path = os.path.join(OUT, slug + ".txt")
        if os.path.exists(path) and os.path.getsize(path) > 400:
            print("skip", slug, flush=True)
            continue
        try:
            d = pull(owner, slug)
        except Exception as e:  # noqa: BLE001
            print("FAIL", slug, e, flush=True)
            continue
        src = d["blob"]["source"]
        nb = json.loads(src) if src.strip().startswith("{") else None
        lines = ["NOTEBOOK %s/%s  (%s)" % (owner, slug, d["metadata"].get("title"))]
        lines.append("kernelVersion %s  enableGpu %s  datasetSources %s modelSources %s" % (
            d["metadata"].get("kernelVersion"), d["metadata"].get("enableGpu"),
            d["metadata"].get("datasetSources"), d["metadata"].get("modelSources")))
        if nb:
            for i, c in enumerate(nb.get("cells", [])):
                body = "".join(c.get("source", []))
                if c.get("cell_type") == "markdown":
                    if body.strip():
                        lines.append("\n[M%d]\n%s" % (i, body.strip()[:2500]))
                else:
                    lines.append("\n[C%d] %d chars\n%s" % (i, len(body), body[:1500]))
        else:
            lines.append(src[:20000])
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("ok", slug, os.path.getsize(path), flush=True)


if __name__ == "__main__":
    main()

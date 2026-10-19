"""Render + push the Kaggle inference kernel for a code competition.

    python build_notebook.py watson [--run kev-0.8b-ft-watson] [--push]

Renders validation/kaggle_notebook/kev_kernel_template.py into
kaggle/kernels/<comp>/{kernel-metadata.json,main.py}. The kernel mounts
  * the public Qwen3.5-0.8B backbone dataset,
  * our private kev bundle (package + heads + adapters),
  * the competition's own data,
so nothing has to be downloaded at run time (code competitions run offline).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATE = HERE / "kaggle_notebook" / "kev_kernel_template.py"
OUT_ROOT = ROOT / "kaggle" / "kernels"

BUNDLE = "ibillxia/kev-bundle"
BASE_DS = "dannybouldiez/qwen35-official-08b-weights"

SPECS = {
    "watson": {
        "competition": "contradictory-my-dear-watson",
        "question": "relation",
        "kind": "choice",
        "instructions": "What is the logical relationship between the premise and the hypothesis?",
        "criteria": ["entailment", "neutral", "contradiction"],
        "label_map": [0, 1, 2],          # the competition's own encoding
        # sample_submission is `id,prediction` with the integer class, not a probability table
        "out_mode": "label",
        "pred_col": "prediction",
        "build_state": (
            "p = (r.get('premise') or '').strip()\n"
            "    h = (r.get('hypothesis') or '').strip()\n"
            "    return 'Premise: %s\\n\\nHypothesis: %s' % (p, h)"
        ),
        "enable_gpu": True,
    },
    "llm": {
        "competition": "llm-classification-finetuning",
        "question": "winner",
        "kind": "choice",
        "instructions": "Which response is better?",
        "criteria": ["winner_model_a", "winner_model_b", "winner_tie"],
        "label_map": [0, 1, 2],
        # scored by log loss, so emit the full probability table (sample_submission has 3 prob columns)
        "out_mode": "probs",
        "pred_col": "",
        # long arena transcripts: keep the head of each field, the state budget is what dominates latency
        # (widths must match what validation/prep_llm.py trained on, or the model sees a different prefix)
        "prompt_chars": 600,
        "resp_chars": 1200,
        "build_state": (
            "def _t(v, w):\n"
            "        try:\n"
            "            x = json.loads(v)\n"
            "            return (' '.join(str(i) for i in x) if isinstance(x, list) else str(x))[:w]\n"
            "        except Exception:\n"
            "            return (v or '')[:w]\n"
            "    return 'Prompt: %s\\n\\nResponse A: %s\\n\\nResponse B: %s' % ("
            "_t(r.get('prompt'), pch), _t(r.get('response_a'), rch), _t(r.get('response_b'), rch))"
        ),
        "enable_gpu": True,
    },
}


# The kernel's field truncation is part of the model's interface: a head trained on 600/1200-char states sees a
# different prefix than one trained on 300/600, and feeding it the wrong width silently costs points. So the width
# travels with the run name instead of sitting in one spec-wide default.
RUN_WIDTHS = {
    "kev-0.8b-llm": (300, 600),        # 12000 rows, max_state 384, 2 ep  -> LB 1.04073
    "kev-0.8b-llm-wide": (600, 1200),  # 20000 rows, max_state 1024, 1 ep -> wider context experiment
}


def render(comp: str, run: str, slug: str, bias_json: str = "", weights: str = ""):
    spec = SPECS[comp]
    runs = [r.strip() for r in run.split(",") if r.strip()]
    wts = [float(w) for w in weights.split(",")] if weights else [1.0] * len(runs)
    if len(wts) != len(runs):
        raise SystemExit("--weights must have one value per --run")
    # class bias fitted by fit_temperature.py --bias (kev's serving path cannot carry it in head.pt)
    class_bias = {}
    bpath = Path(bias_json) if bias_json else (ROOT / "submissions" / f"{runs[0]}.bias.json")
    if bpath.is_file():
        class_bias = json.loads(bpath.read_text(encoding="utf-8")).get("class_bias", {})
        print(f"class bias from {bpath.name}: { {k: round(v, 4) for k, v in class_bias.items()} }")
    # per member: the truncation it was trained with (RUN_WIDTHS) and its weight in the average
    run_spec = []
    for r, w in zip(runs, wts):
        pch, rch = RUN_WIDTHS.get(r, (spec.get("prompt_chars", 300), spec.get("resp_chars", 600)))
        run_spec.append({"run": r, "prompt_chars": pch, "resp_chars": rch, "weight": w})
        print(f"  member {r}: prompt {pch} / response {rch} chars, weight {w}")
    code = TEMPLATE.read_text(encoding="utf-8")
    subs = {
        "__COMP__": spec["competition"],
        "__QUESTION__": spec["question"],
        "__KIND__": spec["kind"],
        "__INSTRUCTIONS__": spec["instructions"],
        "__CRITERIA__": json.dumps(spec["criteria"]).replace('"', '\\"'),
        "__LABEL_MAP__": json.dumps(spec["label_map"]),
        "__OUT_MODE__": spec["out_mode"],
        "__PRED_COL__": spec["pred_col"],
        "__RUN__": runs[0] if len(runs) == 1 else "+".join(runs),
        "__BUILD_STATE__": spec["build_state"],
        "__RUN_SPEC__": json.dumps(run_spec).replace('"', '\\"'),
        "__CLASS_BIAS__": json.dumps(class_bias),
    }
    for k, v in subs.items():
        code = code.replace(k, v)
    out = OUT_ROOT / comp
    out.mkdir(parents=True, exist_ok=True)
    (out / "main.py").write_text(code, encoding="utf-8")
    meta = {
        "id": slug,
        "title": slug.split("/")[-1].replace("-", " "),   # keep id and title in sync or Kaggle re-slugs it
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": spec["enable_gpu"],
        "enable_internet": False,
        "dataset_sources": [BUNDLE, BASE_DS],
        "competition_sources": [spec["competition"]],
        "kernel_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("rendered", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("comp", choices=sorted(SPECS))
    ap.add_argument("--run", default=None, help="run dir name inside the bundle, or a comma-separated list "
                                               "of members to average (probabilities, equal weights unless "
                                               "--weights says otherwise)")
    ap.add_argument("--slug", default=None, help="kaggle kernel id, default ibillxia/kev-<comp>")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--bias-json", default="", help="per-class bias sidecar from fit_temperature.py --bias "
                                                    "(default: submissions/<run>.bias.json when it exists)")
    ap.add_argument("--weights", default="", help="comma-separated member weights, e.g. 2,1 (default equal)")
    a = ap.parse_args()
    run = a.run or SPECS[a.comp].get("default_run", "kev-0.8b-base")
    slug = a.slug or "ibillxia/kev-%s" % a.comp
    out = render(a.comp, run, slug, a.bias_json, a.weights)
    if a.push:
        cli = str(Path(sys.executable).parent / "kaggle.exe")
        r = subprocess.run([cli, "kernels", "push", "-p", str(out)], capture_output=True, text=True)
        print(r.stdout[-1500:], r.stderr[-1500:])


if __name__ == "__main__":
    main()

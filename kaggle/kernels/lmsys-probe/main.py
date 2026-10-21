"""Measure the hidden test set and emit a valid placeholder submission.

For a code competition `test.csv` is a 3-row placeholder in the editor and the real file
only exists at submission time.  The row count decides whether two 9B members can both
be scored inside the 9h cap -- at the measured ~820 tok/s they need ~13h for 25k rows --
so it has to be measured, not guessed.  Kaggle rejects a submission whose kernel never
writes the output file, so this writes the train class prior: a poor score (log loss is
minimised by the prior, ~1.05) but a valid one, and the log carries the answer.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd


def log(*a):
    print(*a, flush=True)


def resolve(sub):
    for pat in ("/kaggle/input/%s" % sub, "/kaggle/input/*/%s" % sub,
                "/kaggle/input/*/*/%s" % sub, "/kaggle/input/*/*/*/%s" % sub):
        hit = sorted(glob.glob(pat))
        if hit:
            return hit[0]
    return None


def find_comp():
    for p in ("/kaggle/input/llm-classification-finetuning",):
        if os.path.isdir(p):
            return p
    for p in sorted(glob.glob("/kaggle/input/*/test.csv")):
        if "llm-classification" in p or "sample_submission" in os.listdir(
                os.path.dirname(p)):
            return os.path.dirname(p)
    for p in sorted(glob.glob("/kaggle/input/*/*/test.csv")):
        return os.path.dirname(p)
    return None


def main():
    log("input tree:", sorted(os.listdir("/kaggle/input")))
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
        if depth > 2:
            dirs[:] = []
            continue
        log("  DIR", root, "| subdirs", dirs[:8], "| files", files[:8])
    for cand in ("/kaggle/input/competitions/llm-classification-finetuning/test.csv",
                 "/kaggle/input/llm-classification-finetuning/test.csv"):
        if os.path.exists(cand):
            log("SIZE", cand, os.path.getsize(cand), "bytes")

    comp = find_comp()
    log("competition dir:", comp)
    if comp is None:
        raise SystemExit("competition dir not found")

    test = pd.read_csv(os.path.join(comp, "test.csv"))
    log("TEST ROWS:", len(test))
    log("test columns:", list(test.columns))
    log("test dtypes:", dict(test.dtypes.astype(str)))
    log("test id head:", test["id"].head(5).tolist())
    ss_path = os.path.join(comp, "sample_submission.csv")
    if os.path.exists(ss_path):
        ss = pd.read_csv(ss_path)
        log("sample_submission rows:", len(ss), "cols:", list(ss.columns))

    # character-length distribution: free, and it pins down the cost of each truncation
    chars = (test["prompt"].astype(str).str.len()
             + test["response_a"].astype(str).str.len()
             + test["response_b"].astype(str).str.len()).values
    log("chars/row mean %.0f p50 %.0f p90 %.0f p99 %.0f max %.0f"
        % (chars.mean(), np.percentile(chars, 50), np.percentile(chars, 90),
           np.percentile(chars, 99), chars.max()))

    # exact token lengths from the real tokenizer (a few MB, no weights needed)
    try:
        from transformers import AutoTokenizer
        p = resolve("lmsys-checkpoints-0-0805")
        log("ckpt dir:", p)
        if p:
            tk = AutoTokenizer.from_pretrained(p)
            idx = np.random.RandomState(0).permutation(len(test))[:2000]
            lens = []
            for i in idx:
                r = test.iloc[i]
                lens.append(len(tk(str(r["prompt"]) + str(r["response_a"])
                                   + str(r["response_b"]))["input_ids"]))
            lens = np.array(lens)
            log("raw token len (sample %d) mean %.0f p50 %.0f p90 %.0f p99 %.0f max %.0f"
                % (len(lens), lens.mean(), np.percentile(lens, 50),
                   np.percentile(lens, 90), np.percentile(lens, 99), lens.max()))
            for L in (768, 1024, 1536, 2048, 3072, 4096):
                log("  L=%4d -> tok/row %.0f (%.0f%% of the 2048 baseline)"
                    % (L, np.minimum(lens, L).mean(),
                       100.0 * np.minimum(lens, L).mean() / max(np.minimum(lens, 2048).mean(), 1)))
    except Exception as e:  # noqa: BLE001
        log("tokenizer probe failed:", repr(e))

    # train class prior: the log-loss-optimal constant, so this placeholder is not
    # merely valid but actually the best constant predictor available
    prior = np.array([0.5, 0.5, 0.0])
    try:
        tr = pd.read_csv(os.path.join(comp, "train.csv"), usecols=[
            "winner_model_a", "winner_model_b", "winner_tie"])
        cnt = tr.values.sum(0).astype(float)
        prior = np.clip(cnt / cnt.sum(), 1e-3, None)
        prior = prior / prior.sum()
        log("train rows %d, class counts %s, prior %s"
            % (len(tr), cnt.astype(int).tolist(), np.round(prior, 4).tolist()))
    except Exception as e:  # noqa: BLE001
        log("train prior failed, falling back to uniform-ish:", repr(e))

    out = pd.DataFrame({
        "id": test["id"].values,
        "winner_model_a": np.full(len(test), prior[0]),
        "winner_model_b": np.full(len(test), prior[1]),
        "winner_tie": np.full(len(test), prior[2]),
    })
    out.to_csv("submission.csv", index=False)
    log("wrote submission.csv", out.shape)
    log("PROBE DONE")


if __name__ == "__main__":
    main()

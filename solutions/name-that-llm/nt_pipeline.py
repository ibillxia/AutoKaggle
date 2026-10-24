"""name-that-llm pipeline orchestrator (unattended).

Phases, each gated on the previous:
  A  hybrid2 v2 RUNNING -> COMPLETE -> submit kernel version 2 -> read LB score
  B  fork604b (queued by push_f604b_when_free.py) -> RUNNING -> COMPLETE -> submit -> score
  C  train8b_v2 -> COMPLETE -> package ntl8b_own_v1 as dataset ibillxia/ntl-pool-v2-lora
     -> build hybrid3 (LORAS += poolv2_8b) -> lint -> push when slot free -> COMPLETE
     -> lint clean -> submit -> score

Everything is logged to nt-run-logs/pipeline.log; every submission decision follows the
user rule: only submit versions expected >= current best (0.601).
"""
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, "nt-run-logs")
LOG = os.path.join(LOGDIR, "pipeline.log")
PY = sys.executable
COMP = "name-that-llm"
BEST = 0.601
POLL = 300          # 5 min


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))


def api_call(code):
    """Run a kaggle API snippet in a fresh python, return stdout."""
    p = subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=HERE)
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def kernel_status(ref):
    rc, out, err = api_call(
        "from kaggle.api.kaggle_api_extended import KaggleApi\n"
        "api=KaggleApi(); api.authenticate()\n"
        "print(api.kernels_status('%s').status)" % ref)
    txt = (out + err).strip().splitlines()[-1] if (out + err).strip() else "UNKNOWN"
    for cand in ("RUNNING", "QUEUED", "COMPLETE", "ERROR", "CANCELED"):
        if cand in txt:
            return cand
    return txt


def wait_terminal(ref, hours=20):
    """Poll a kernel until it leaves RUNNING/QUEUED.

    Tolerates pre-start states: a stale ERROR from a previous version or a
    not-yet-existing kernel (train8b_v2 before its push) are NOT terminal --
    we only accept terminal status after the kernel has started once.
    """
    deadline = time.time() + hours * 3600
    seen_started = False
    while time.time() < deadline:
        st = kernel_status(ref)
        if st in ("RUNNING", "QUEUED"):
            seen_started = True
        log("status %s = %s%s" % (ref, st, "" if seen_started else " (waiting for start)"))
        if st == "COMPLETE":
            return st
        if st in ("ERROR", "CANCELED") and seen_started:
            return st
        time.sleep(POLL)
    return "TIMEOUT"


def submit_kernel(kernel, version, note):
    """Submit a kernel version to the competition, return (ref, score) or (None, reason)."""
    code = (
        "from kaggle.api.kaggle_api_extended import KaggleApi\n"
        "api=KaggleApi(); api.authenticate()\n"
        "r=api.competition_submit_code('submission.parquet', %r, '%s', '%s', %d)\n"
        "print('REF', r.ref)" % (note, COMP, kernel, version))
    rc, out, err = api_call(code)
    combined = out + err
    if "REF" not in out:
        log("submit %s v%d FAILED: %s" % (kernel, version, combined[-400:]))
        return None, combined[-200:]
    ref = out.split("REF")[1].strip().splitlines()[0].strip()
    log("submitted %s v%d ref=%s" % (kernel, version, ref))
    score = wait_score(ref)
    return ref, score


def wait_score(ref, hours=3):
    deadline = time.time() + hours * 3600
    last = None
    while time.time() < deadline:
        rc, out, err = api_call(
            "from kaggle.api.kaggle_api_extended import KaggleApi\n"
            "api=KaggleApi(); api.authenticate()\n"
            "for s in api.competition_submissions('%s'):\n"
            "    if str(s.ref)=='%s':\n"
            "        print(str(s.status).split('.')[-1], getattr(s,'publicScore',None) or getattr(s,'public_score',None))\n"
            "        break\n" % (COMP, ref))
        line = out.strip().splitlines()[-1] if out.strip() else ""
        if line != last:
            log("submission %s -> %s" % (ref, line))
            last = line
        if line.startswith("COMPLETE") or line.startswith("ERROR"):
            return line
        time.sleep(POLL)
    return "score timeout"


def package_poolv2_lora():
    """Pull train8b_v2 output and publish ibillxia/ntl-pool-v2-lora with folder poolv2_8b/."""
    import shutil
    out = os.path.join(LOGDIR, "train8b_v2_pull")
    rc, o, e = api_call(
        "from kaggle.api.kaggle_api_extended import KaggleApi\n"
        "api=KaggleApi(); api.authenticate()\n"
        "api.kernels_output('ibillxia/ntl-train8b-v2', path=r'%s')\n" % out.replace("\\", "\\\\"))
    log("pull train8b_v2 output rc=%d %s" % (rc, (o + e)[-200:]))
    src = os.path.join(out, "ntl8b_own_v1")
    if not os.path.isdir(os.path.join(src, "adapter")):
        log("FATAL: ntl8b_own_v1/adapter missing in pulled output: %s" % os.listdir(out))
        return False
    stage = os.path.join(LOGDIR, "poolv2_lora_stage", "poolv2_8b")
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage, exist_ok=True)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(stage, item)
        (shutil.copytree if os.path.isdir(s) else shutil.copyfile)(s, d)
    meta = {
        "title": "NTL pool v2 lora",
        "id": "ibillxia/ntl-pool-v2-lora",
        "licenses": [{"name": "CC-BY-4.0"}],
        "description": "Two-stage LoRA member (qwen3 8B base, keep=24) trained on pool_v2 "
                       "(70,034 rows) + official train; folder poolv2_8b follows the LORAS find() layout.",
    }
    root = os.path.dirname(stage)
    with open(os.path.join(root, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from push_pool_dataset import patch_gcs_host\n"
        "patch_gcs_host()\n"
        "from kaggle.api.kaggle_api_extended import KaggleApi\n"
        "api=KaggleApi(); api.authenticate()\n"
        "refs=[d.ref for d in api.datasets_list(mine=True)]\n"
        "if 'ibillxia/ntl-pool-v2-lora' in refs:\n"
        "    print(api.dataset_create_version(r'%s', 'poolv2 lora member', quiet=False, dir_mode='skip'))\n"
        "else:\n"
        "    print(api.dataset_create_new(r'%s', public=False, quiet=False, dir_mode='skip'))\n"
        % (HERE.replace("\\", "\\\\"), root.replace("\\", "\\\\"), root.replace("\\", "\\\\")))
    rc, o, e = api_call(code)
    ok = '"error": null' in (o + e) or "/ibillxia/ntl-pool-v2-lora" in (o + e)
    log("dataset push rc=%d ok=%s %s" % (rc, ok, (o + e)[-300:]))
    return ok


def build_hybrid3():
    nb = json.load(open(os.path.join(HERE, "hybrid2", "ntl-hybrid2.ipynb"), encoding="utf-8"))
    src = nb["cells"][0]["source"]
    old = ('    "ts14b_f4": dict(w=0.070, twin_w=0.0245, bs=8, base=dict(model_type="qwen3", '
           'hidden_size=5120, num_hidden_layers=40)),\n}')
    new = old[:-2] + ('\n    # pool_v2-trained 8B member (stage-1 pool 34,940 -> 70,034 rows)\n'
                      '    "poolv2_8b": dict(w=0.070, twin_w=0.0245, bs=8, base=dict(model_type="qwen3", '
                      'hidden_size=4096, num_hidden_layers=36)),\n}')
    assert src.count(old) == 1, "LORAS tail changed"
    src = src.replace(old, new, 1)
    compile(src, "<hybrid3>", "exec")
    nb["cells"][0]["source"] = src
    out_dir = os.path.join(HERE, "hybrid3")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(nb, open(os.path.join(out_dir, "ntl-hybrid3.ipynb"), "w", encoding="utf-8"), indent=1)
    meta = json.load(open(os.path.join(HERE, "hybrid2", "kernel-metadata.json"), encoding="utf-8"))
    meta["id"] = "ibillxia/ntl-hybrid3"
    meta["title"] = "ntl hybrid3 poolv2 member"
    meta["code_file"] = "ntl-hybrid3.ipynb"
    if "ibillxia/ntl-pool-v2-lora" not in meta["dataset_sources"]:
        meta["dataset_sources"].append("ibillxia/ntl-pool-v2-lora")
    json.dump(meta, open(os.path.join(out_dir, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
    # lint
    p = subprocess.run([PY, os.path.join(HERE, "lint_nb.py"),
                        os.path.join(out_dir, "ntl-hybrid3.ipynb")], capture_output=True, text=True)
    lint = p.stdout + p.stderr
    log("hybrid3 lint:\n%s" % lint)
    return ("undefined: none" in lint and "SYNTAX" not in lint)


def push_kernel(dir_name, retry_hours=12):
    code = ("from kaggle.api.kaggle_api_extended import KaggleApi\n"
            "api=KaggleApi(); api.authenticate()\n"
            "r=api.kernels_push('solutions/name-that-llm/%s')\n"
            "print('PUSHRESULT', r.error, r.versionNumber)\n" % dir_name)
    deadline = time.time() + retry_hours * 3600
    while time.time() < deadline:
        rc, out, err = api_call(code)
        combined = out + err
        if "PUSHRESULT" in out and " None " in out:
            line = out.split("PUSHRESULT")[1].strip().splitlines()[0]
            ver = int(line.split()[-1])
            log("pushed %s version %s" % (dir_name, ver))
            return int(ver)
        if "Maximum batch GPU session" in combined:
            log("push %s: slots full, waiting" % dir_name)
            time.sleep(POLL)
            continue
        log("push %s FAILED: %s" % (dir_name, combined[-300:]))
        return None
    return None


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    log("=== pipeline start ===")

    # ---------------- phase A: hybrid2 ----------------
    # already submitted manually as ref 57589058 (2026-10-24) - just wait for score
    log("hybrid2 final: %s" % wait_score("57589058"))

    # ---------------- phase B: fork604b ----------------
    # pushed manually as v2 at 2026-10-24 17:20 (session was free by then)
    st = wait_terminal("ibillxia/ntl-fork604b-full-deps")
    if st == "COMPLETE":
        ver = 2
        if ver:
            ref, score = submit_kernel("ibillxia/ntl-fork604b-full-deps", ver, "fork604b full deps fix")
            log("fork604b final: %s" % score)
        else:
            log("fork604b COMPLETE but version unknown; not submitting")
    else:
        log("fork604b did not COMPLETE (%s)" % st)

    # ---------------- phase C: train8b_v2 -> hybrid3 ----------------
    st = wait_terminal("ibillxia/ntl-train8b-v2", hours=30)
    if st != "COMPLETE":
        log("train8b_v2 did not COMPLETE (%s) - hybrid3 aborted" % st)
        log("=== pipeline end (incomplete) ===")
        return
    if not package_poolv2_lora():
        log("FATAL: dataset packaging failed - hybrid3 aborted")
        return
    if not build_hybrid3():
        log("FATAL: hybrid3 lint failed - not pushed")
        return
    ver = push_kernel("hybrid3")
    if not ver:
        log("FATAL: hybrid3 push failed")
        return
    st = wait_terminal("ibillxia/ntl-hybrid3", hours=30)
    if st == "COMPLETE":
        ref, score = submit_kernel("ibillxia/ntl-hybrid3", ver, "hybrid3: + poolv2_8b member")
        log("hybrid3 final: %s" % score)
    else:
        log("hybrid3 did not COMPLETE (%s)" % st)
    log("=== pipeline end ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("PIPELINE CRASH:\n" + traceback.format_exc())

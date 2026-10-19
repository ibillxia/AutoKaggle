#!/usr/bin/env python
"""Run one targeted test with a bounded timeout and record the outcome.

    python run_targeted_test.py --workspace /workspace --test tests/test_x.py

Writes <workspace>/.repro/repro_log.json. Always exits 0: the point is to
*record* a failure, not to fail the agent's turn. The agent reads the JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TAIL_CHARS = 2000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="/workspace")
    ap.add_argument("--test", required=True, help="test file or node id")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)

    ws = Path(args.workspace)
    if not ws.is_dir():
        print(f"workspace not found: {ws}", file=sys.stderr)
        return 0

    cmd = [args.python, "-m", "pytest", args.test, "-q", "--no-header", "-x"]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(ws), capture_output=True, text=True,
            timeout=args.timeout,
        )
        returncode, out, err = proc.returncode, proc.stdout, proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode, timed_out = 124, True
        out = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    except FileNotFoundError as exc:
        returncode, timed_out = 127, False
        out, err = "", str(exc)

    duration = round(time.time() - started, 3)
    passed = returncode == 0
    tail = (out + err)[-TAIL_CHARS:]

    record = {
        "command": " ".join(cmd),
        "test": args.test,
        "returncode": returncode,
        "passed": passed,
        "timed_out": timed_out,
        "duration_s": duration,
        "tail": tail,
    }

    out_dir = ws / ".repro"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "repro_log.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8",
    )

    verdict = "PASSED" if passed else ("TIMED OUT" if timed_out else "FAILED")
    print(f"[repro_first] {args.test}: {verdict} (rc={returncode}, {duration}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

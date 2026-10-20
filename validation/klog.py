"""Pull a Kaggle kernel's log and print it as plain text.

The .log file Kaggle hands back is a JSON stream split into tiny chunks with escaped
newlines, which is unreadable raw.  This unwraps it and prints the tail.

    python klog.py <owner/slug> [lines] [--err]
"""
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

KAGGLE = r"C:/Users/deepg/.workbuddy/binaries/python/envs/default/Scripts/kaggle.exe"


def main():
    slug = sys.argv[1]
    tail = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 4000
    only_err = "--err" in sys.argv
    out = tempfile.mkdtemp()
    subprocess.run([KAGGLE, "kernels", "output", slug, "-p", out],
                   capture_output=True, text=True)
    logs = glob.glob(os.path.join(out, "*.log"))
    if not logs:
        print("no log at", out)
        return
    raw = open(logs[0], encoding="utf-8").read()
    txt = ""
    for m in re.finditer(r'"data":"((?:[^"\\]|\\.)*)"', raw):
        txt += json.loads('"' + m.group(1) + '"')
    if only_err:
        lines = [l for l in txt.splitlines()
                 if l.startswith("  File") or "Error" in l or "error" in l or "Traceback" in l]
        txt = "\n".join(lines)
    print(txt[-tail:])


if __name__ == "__main__":
    main()

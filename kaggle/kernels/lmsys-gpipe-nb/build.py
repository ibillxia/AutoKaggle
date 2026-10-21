"""Embed worker_body.py into main.py so the two can never drift apart again.

Kaggle only ships main.py (plus the notebook), so the worker has to live inside
it as a string literal. Editing both copies by hand already cost us one silent
regression (the cfg_path fix got lost during a rebuild), hence this script.

Run:  python build.py        # rewrites the WORKER_SRC block, then the notebook
      python build.py --check  # fail if the block is stale
"""
import ast
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BODY = os.path.join(HERE, "worker_body.py")
MAIN = os.path.join(HERE, "main.py")
NB = os.path.join(HERE, "lmsys-gpipe.ipynb")
PAT = re.compile(r'WORKER_SRC = r?""".*?"""(?=\s*\n\s*\n\s*def sm\()', re.S)


def notebook():
    """Kaggle's entry point is the notebook, which is just main.py in one cell."""
    code = io.open(MAIN, encoding="utf-8").read()
    nb = {
        "cells": [{"cell_type": "code", "execution_count": None,
                   "metadata": {}, "outputs": [], "source": code}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python", "version": "3.13"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with io.open(NB, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f)
    print("wrote %s (%d bytes)" % (os.path.basename(NB), len(code)))


def main():
    src = io.open(BODY, encoding="utf-8").read().rstrip("\n") + "\n"
    ast.parse(src)                                   # the worker must be valid python
    if '"""' in src:
        # the block is a triple-quoted literal, so a docstring in the worker closes it
        sys.exit('worker_body.py contains a """ docstring; use # comments instead')

    cur = io.open(MAIN, encoding="utf-8").read()
    m = PAT.search(cur)
    if not m:
        sys.exit("WORKER_SRC block not found in main.py")

    block = 'WORKER_SRC = r"""' + src + '"""'
    if "--check" in sys.argv:
        if m.group(0) != block:
            sys.exit("WORKER_SRC is stale - run build.py")
        print("WORKER_SRC up to date (%d bytes, %d lines)"
              % (len(src), src.count("\n")))
        return

    out = cur[:m.start()] + block + cur[m.end():]
    io.open(MAIN, "w", encoding="utf-8", newline="\n").write(out)
    ast.parse(out)                                   # the host must stay valid too
    print("embedded %d bytes / %d lines into main.py" % (len(src), src.count("\n")))
    notebook()


if __name__ == "__main__":
    main()

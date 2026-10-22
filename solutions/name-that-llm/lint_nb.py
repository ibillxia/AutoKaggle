"""Crude undefined-name lint for a notebook's joined code cells.

Kaggle kernels fail at runtime with NameError only after hours of GPU work, so
catch the cheap class of bugs locally: a Name that is loaded somewhere but never
bound anywhere in the notebook.  Not scope-aware (function locals are treated as
global), which is fine because a false negative here is what we care about --
every reported name is genuinely never assigned in any cell.
"""
import ast
import builtins
import json
import sys


def names_of(tree):
    bound, loaded = set(dir(builtins)), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name):
            (bound.add(node.id) if isinstance(node.ctx, (ast.Store, ast.Del)) else loaded.add(node.id))
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Import):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            bound.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global):
            bound.update(node.names)
    return bound, loaded


for path in sys.argv[1:]:
    nb = json.load(open(path, encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    bound, loaded, bad_syntax = set(dir(builtins)), set(), []
    for i, src in enumerate(cells):
        try:
            b, l = names_of(ast.parse(src))
        except SyntaxError as e:
            bad_syntax.append(f"cell{i}: {e}")
            continue
        bound |= b
        loaded |= l
    missing = sorted(loaded - bound)
    print(f"{path}: {len(cells)} code cells")
    for m in bad_syntax:
        print("  SYNTAX", m)
    print("  undefined:", missing if missing else "none")

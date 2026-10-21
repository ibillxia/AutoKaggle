import os
def walk(base, depth=4):
    for root, dirs, files in os.walk(base):
        rel = os.path.relpath(root, base)
        if rel.count(os.sep) >= depth:
            dirs[:] = []
            continue
        print(" DIR", rel, "->", sorted(files)[:10])
print("### /kaggle/input")
walk("/kaggle/input", 4)
print("### done")

import os, json, glob
print("### /kaggle/input top:", sorted(os.listdir("/kaggle/input")))
for root, dirs, files in os.walk("/kaggle/input"):
    rel = os.path.relpath(root, "/kaggle/input")
    if rel.count(os.sep) > 2: dirs[:] = []; continue
    print(" DIR", rel, "->", sorted(files)[:14])
cfg = glob.glob("/kaggle/input/**/config.json", recursive=True)
for c in cfg:
    d = json.load(open(c))
    tc = d.get("text_config", {})
    print("CONFIG", c, "| model_type", d.get("model_type"), "| hidden", tc.get("hidden_size"), "| layers", len(tc.get("layer_types", [])))
print("### done")

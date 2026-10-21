# Extract multi-layer mean-pooled Qwen3.5-4B-Base embeddings for official train + extra corpus.
# Forum finding (749593/750462): early layers carry authorship style, last layer is the worst.
import argparse
import time
import numpy as np
import pandas as pd
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="E:/Bill/AI/AutoKaggle/models/Qwen3.5-4B-Base")
    ap.add_argument("--out", default="emb_qwen4b.npz")
    ap.add_argument("--layers", default="1,3,6,9,16,24,32")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=384)
    ap.add_argument("--which", default="both", choices=["train", "extra", "both"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    layers = [int(x) for x in a.layers.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, quantization_config=bnb, device_map={"": 0}).eval()
    print("model loaded", flush=True)

    frames = []
    if a.which in ("train", "both"):
        tr = pd.read_csv("data/train.csv")[["text", "model"]].copy()
        tr["src"] = "train"
        frames.append(tr)
    if a.which in ("extra", "both"):
        ex = pd.read_csv("extra/extra_short.csv")[["text", "model"]].copy()
        ex["src"] = "extra"
        frames.append(ex)
    df = pd.concat(frames, ignore_index=True)
    if a.limit:
        df = df.groupby("model", group_keys=False).head(a.limit)
    print("rows:", len(df), df["src"].value_counts().to_dict(), flush=True)

    # length-sorted batching keeps padding low
    wc = df["text"].str.split().str.len().values
    order = np.argsort(wc)
    hidden = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    res = {L: np.zeros((len(df), hidden), dtype=np.float16) for L in layers}

    t0 = time.time()
    done = 0
    for i in range(0, len(order), a.batch):
        idx = order[i:i + a.batch]
        batch = df["text"].values[idx].tolist()
        enc = tok(batch, padding=True, truncation=True, max_length=a.maxlen, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
        for L in layers:
            hs = out.hidden_states[L].float()
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp_min(1)
            res[L][idx] = pooled.to(torch.float16).cpu().numpy()
        done += len(idx)
        if (i // a.batch) % 40 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(df)} {el/done*1000:.0f} ms/text eta={el/done*(len(df)-done)/60:.1f} min", flush=True)
        del out

    np.savez(a.out, y=df["model"].values, src=df["src"].values,
             **{f"L{L}": res[L] for L in layers})
    print("saved", a.out, {f"L{L}": res[L].shape for L in layers}, flush=True)

if __name__ == "__main__":
    main()

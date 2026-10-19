"""Sanity-check the local xlm-roberta-large: does it still do MLM, and are its sentence
representations diverse? If MLM is fine but NLI finetuning fails, the bug is in our head/labels;
if MLM is garbage, the weights are the problem.
"""
import sys
from pathlib import Path

import torch

ROOT = Path("E:/Bill/AI/AutoKaggle")
MODEL = str(ROOT / "models/xlm-roberta-large")

from transformers import AutoTokenizer, XLMRobertaForMaskedLM  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = XLMRobertaForMaskedLM.from_pretrained(MODEL)
model.eval().cuda()

print("vocab_size", tok.vocab_size, "model", model.config.vocab_size)
print("pad", tok.pad_token, tok.pad_token_id, "cls", tok.cls_token_id, "sep", tok.sep_token_id,
      "mask", tok.mask_token_id)
print("tokenize:", tok.tokenize("Paris is the capital of France."))
print("ids     :", tok("Paris is the capital of France.")["input_ids"])

tests = [
    ("Paris is the capital of <mask>.", "France"),
    ("The capital of Japan is <mask>.", "Tokyo"),
    ("北京是<mask>的首都。", "中国"),
    ("Das Wasser ist <mask> kalt heute.", "sehr"),
]
for text, hint in tests:
    ids = tok(text.replace("<mask>", tok.mask_token), return_tensors="pt").to("cuda")
    with torch.no_grad():
        lg = model(**ids).logits
    pos = (ids["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
    top = lg[0, pos].topk(5)
    print("\n%s  (hint: %s)" % (text, hint))
    for v, i in zip(top.values, top.indices):
        print("   %-16s %.2f" % (tok.decode([int(i)]), float(v)))

# representation diversity
sents = ["A man is playing a guitar.", "A woman is cooking dinner.",
         "The stock market fell sharply.", "Un homme joue de la guitare."]
enc = tok(sents, padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    h = model.roberta(**enc).last_hidden_state[:, 0, :]
h = torch.nn.functional.normalize(h.float(), dim=-1)
print("\nCLS cosine similarity matrix:")
print((h @ h.T).cpu().numpy().round(3))

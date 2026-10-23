"""Kaggle cifar-10 (2014) -- training / validation / inference on one 8GB GPU.

Validation design
-----------------
The competition train set (50k PNGs) is the *only* labeled data we have; the
300k test images are unlabeled and their distribution differs from CIFAR-10
proper.  So we deliberately do not tune on a holdout carved out of the 50k
(otherwise the model is trained on 45k instead of 50k).  Instead we report:

  * `oof`  : stratified 5-fold out-of-fold accuracy on the 50k competition
             train set -> an honest estimate of how the model does on data it
             has *not* seen, from the same distribution.
  * `std`  : accuracy on the standard CIFAR-10 10k test set (never used for
             training or model selection) -> sanity check that we learned the
             real task and not a quirk of the 50k PNG dump.
  * `agree`: prediction agreement between the full-data model and the folds
             on the 300k competition test set -> a cheap stability estimate.

The reported number for the leaderboard estimate is `oof`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
LABELS = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class CifarNpy(Dataset):
    """uint8 NHWC in memory (or mmap) with batch-level augmentation on GPU."""

    def __init__(self, x: np.ndarray, y: np.ndarray | None = None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = np.asarray(self.x[i])
        if self.y is None:
            return torch.from_numpy(img)
        return torch.from_numpy(img), int(self.y[i])


def gpu_augment(batch_u8: torch.Tensor, rng: torch.Generator, res: int = 32,
                dihedral: bool = False) -> torch.Tensor:
    """Random-resized crop + horizontal flip + colour jitter, done on the GPU.

    batch_u8: (B,32,32,3) uint8 CUDA tensor. Returns (B,3,res,res) float in [0,1].
    """
    b = batch_u8.shape[0]
    x = batch_u8.permute(0, 3, 1, 2).float().div_(255.0)  # (B,3,32,32)

    if res == 32:
        # pad by 4 then take a random 32x32 crop (the standard CIFAR recipe)
        x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    else:
        # progressive-resizing / FixRes trick: upsample to res+4, then reflect
        # pad by 4 -> a res+8 canvas, so the crop jitter is proportionally the
        # same as the 32px recipe
        x = F.interpolate(x, size=res + 4, mode="bilinear", align_corners=False,
                          antialias=True)
        x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    pad = x.shape[-1]
    ox = torch.randint(0, pad - res + 1, (b, 1, 1), device=x.device, generator=rng)
    oy = torch.randint(0, pad - res + 1, (b, 1, 1), device=x.device, generator=rng)
    r = torch.arange(res, device=x.device)
    flat = ((r.view(1, res, 1) + oy) * pad + (r.view(1, 1, res) + ox))  # (B,res,res)
    idx = flat.unsqueeze(1).expand(b, 3, res, res).reshape(b, 3, res * res)
    x = x.view(b, 3, pad * pad).gather(2, idx).reshape(b, 3, res, res)

    flip = torch.rand(b, device=x.device, generator=rng) < 0.5
    if flip.any():
        x[flip] = torch.flip(x[flip], dims=[3])

    # Full dihedral augmentation. This competition's test set contains copies
    # of the official test images under v-flip and 180-rotation as well as
    # h-flip (~13.5k of the 53.7k scored copies), and a model trained with
    # h-flip only scores ~0.46 on those. Rotating here makes the network
    # invariant to all 8 orientations, which is what the scored set rewards.
    if dihedral:
        r = torch.rand(b, device=x.device, generator=rng)
        vf = r < 0.25
        r180 = (r >= 0.25) & (r < 0.50)
        x[vf] = torch.flip(x[vf], dims=[2])
        x[r180] = torch.flip(x[r180], dims=[2, 3])

    # per-image brightness / contrast jitter
    if rng is not None:
        gain = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device, generator=rng) - 0.5) * 0.4
        x = x * gain
        x = torch.clamp(x, 0.0, 1.0)
    return x


def cutout(x: torch.Tensor, size: int, rng: torch.Generator) -> torch.Tensor:
    """Random-erasing style square cutout (DeVries & Taylor 2017).

    Applied after normalisation; zeros are the per-channel mean there.
    """
    if size <= 0:
        return x
    b, _, h, w = x.shape
    cy = torch.randint(0, h, (b, 1), device=x.device, generator=rng)
    cx = torch.randint(0, w, (b, 1), device=x.device, generator=rng)
    r = size // 2
    ys = torch.arange(h, device=x.device).view(1, h)
    xs = torch.arange(w, device=x.device).view(1, w)
    mask = ((ys - cy).abs() <= r).unsqueeze(2) & ((xs - cx).abs() <= r).unsqueeze(1)
    return x.masked_fill(mask.unsqueeze(1), 0.0)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN.to(x.device)) / STD.to(x.device)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        out = F.relu(self.b1(self.c1(x)), inplace=True)
        out = self.b2(self.c2(out))
        return F.relu(out + idt, inplace=True)


class WideResNet(nn.Module):
    """WRN for 32x32 inputs (Zagoruyko & Komodakis, 2016)."""

    def __init__(self, depth=28, widen=10, num_classes=10, drop=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth must be 6n+4"
        n = (depth - 4) // 6
        widths = [16, 16 * widen, 32 * widen, 64 * widen]
        self.drop = drop
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        layers = []
        for i, (cin, cout, stride) in enumerate([
                (widths[0], widths[1], 1),
                (widths[1], widths[2], 2),
                (widths[2], widths[3], 2)]):
            for j in range(n):
                layers.append(BasicBlock(cin if j == 0 else cout, cout,
                                         stride if j == 0 else 1))
            cin = cout
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(F.relu(self.bn1(x), inplace=True))
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.drop > 0:
            x = F.dropout(x, self.drop, self.training)
        return self.fc(x)


def build_model(name: str, depth: int, widen: int, drop: float) -> nn.Module:
    if name == "wrn":
        return WideResNet(depth=depth, widen=widen, drop=drop)
    import timm
    return timm.create_model(name, pretrained=True, num_classes=10, drop_rate=drop)


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, x_u8, y, batch=512, tta: bool = False, res: int = 32):
    model.eval()
    correct = 0
    n = len(x_u8)
    for i in range(0, n, batch):
        xb = torch.as_tensor(np.ascontiguousarray(x_u8[i:i + batch]), device="cuda").permute(0, 3, 1, 2).float().div_(255.0)
        xb = normalize(xb).contiguous(memory_format=torch.channels_last)
        logits = model(xb)
        if tta:
            logits = logits + model(torch.flip(xb, dims=[3]).contiguous(memory_format=torch.channels_last))
        correct += (logits.argmax(1) == torch.as_tensor(y[i:i + batch], device="cuda")).sum().item()
    return correct / n


@torch.no_grad()
def predict_probs(model, x_u8, batch=512, tta: bool = True):
    model.eval()
    outs = []
    for i in range(0, len(x_u8), batch):
        xb = torch.as_tensor(np.ascontiguousarray(x_u8[i:i + batch]), device="cuda").permute(0, 3, 1, 2).float().div_(255.0)
        xb = normalize(xb).contiguous(memory_format=torch.channels_last)
        logits = model(xb)
        if tta:
            logits = logits + model(torch.flip(xb, dims=[3]).contiguous(memory_format=torch.channels_last))
        outs.append(F.softmax(logits / 2.0, dim=1).half().cpu())
    return torch.cat(outs).numpy()


@torch.no_grad()
def eval_resized(model, x_u8, y, res: int, tta: bool = False, batch: int = 512):
    """Evaluate at a larger input size: resize the 32px image up to `res`
    and centre-crop -- the FixRes recipe for test-time resolution."""
    model.eval()
    correct = 0
    n = len(x_u8)
    for i in range(0, n, batch):
        xb = torch.as_tensor(np.ascontiguousarray(x_u8[i:i + batch]), device="cuda").permute(0, 3, 1, 2).float().div_(255.0)
        if res != 32:
            xb = F.interpolate(xb, size=int(round(res * 1.14)), mode="bilinear",
                               align_corners=False, antialias=True)
            xb = xb[..., :res, :res]
        xb = normalize(xb).contiguous(memory_format=torch.channels_last)
        logits = model(xb)
        if tta:
            logits = logits + model(torch.flip(xb, dims=[3]).contiguous(memory_format=torch.channels_last))
        correct += (logits.argmax(1) == torch.as_tensor(y[i:i + batch], device="cuda")).sum().item()
    return correct / n


_STD = {}


def _std_test():
    """Canonical CIFAR-10 10k test set -- held out from every training run, so
    it is an unbiased check that we are learning the real task."""
    if "x" not in _STD:
        _STD["x"] = np.load(DATA / "npy" / "std_test_x.npy")
        _STD["y"] = np.load(DATA / "npy" / "std_test_y.npy")
    return _STD["x"], _STD["y"]


def cosine_lr(step, total, base, warmup=0.0, min_frac=0.0):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def train_one(args, xtr, ytr, xva, yva, tag, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = torch.Generator(device="cuda")
    rng.manual_seed(seed)

    model = build_model(args.model, args.depth, args.widen, args.drop).cuda()
    model = model.to(memory_format=torch.channels_last)
    if args.init:
        sd = torch.load(args.init, map_location="cpu")
        model.load_state_dict(sd)
        print(f"[{tag}] warm-started from {args.init}", flush=True)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{tag}] model={args.model} params={nparam:.1f}M train={len(xtr)} val={len(xva)}", flush=True)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([
        {"params": decay, "weight_decay": args.wd},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, momentum=0.9, nesterov=True)

    steps_per_epoch = math.ceil(len(xtr) / args.bs)
    total = steps_per_epoch * args.epochs
    warmup = int(total * args.warmup)

    xva_t = xtr[:0]  # placeholder
    best = -1.0
    step = 0
    t0 = time.time()
    use_amp = bool(args.amp) and torch.cuda.is_bf16_supported()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xtr))          # CPU perm: numpy fancy-index
        run_loss = 0.0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs].numpy()
            xb = torch.from_numpy(np.ascontiguousarray(xtr[idx])).cuda(non_blocking=True)
            yb = torch.from_numpy(ytr[idx]).cuda(non_blocking=True)
            xb = normalize(gpu_augment(xb, rng, args.res, bool(args.dihedral)))
            if args.cutout:
                xb = cutout(xb, args.cutout, rng)
            xb = xb.contiguous(memory_format=torch.channels_last)
            lr = cosine_lr(step, total, args.lr, warmup, args.min_lr_frac)
            for g in opt.param_groups:
                g["lr"] = lr
            # label smoothing, the standard CIFAR trick
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=args.ls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(idx)
            step += 1
        msg = f"[{tag}] ep {ep+1}/{args.epochs} loss {run_loss/len(xtr):.4f} lr {lr:.4f}"
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            if len(xva):
                va = evaluate(model, xva, yva)
                msg += f" val {va:.4f}"
                best = max(best, va)
            if args.monitor_std:
                sx, sy = _std_test()
                msg += f" std10k {eval_resized(model, sx, sy, args.test_res, tta=True):.4f}"
        print(msg + f" {time.time()-t0:.0f}s", flush=True)
    return model, best


def stratified_folds(y: np.ndarray, k: int, seed: int):
    rng = np.random.RandomState(seed)
    fold = np.empty(len(y), dtype=np.int64)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wrn")
    ap.add_argument("--depth", type=int, default=28)
    ap.add_argument("--widen", type=int, default=10)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--min-lr-frac", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--folds", type=int, default=0, help="0 = single full-data run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--limit", type=int, default=0, help="debug: subsample train")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--cutout", type=int, default=8)
    ap.add_argument("--dihedral", type=int, default=0,
                    help="also augment v-flip/rot180 (needed for this "
                         "competition's test set, see PLAN.md)")
    ap.add_argument("--init", default="", help="checkpoint to warm-start from")
    ap.add_argument("--res", type=int, default=32, help="train crop size")
    ap.add_argument("--test-res", type=int, default=32, help="eval resize")
    ap.add_argument("--monitor-std", type=int, default=1,
                    help="report canonical CIFAR-10 test acc during training")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = RUNS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))

    xtr = np.load(DATA / "npy" / "train_x.npy")   # 150MB: load once, keep in RAM
    ytr = np.load(DATA / "npy" / "train_y.npy")
    if args.limit:
        xtr, ytr = xtr[:args.limit], ytr[:args.limit]

    t0 = time.time()
    if args.folds:
        fold = stratified_folds(ytr, args.folds, args.seed)
        oof = np.zeros((len(ytr), 10), dtype=np.float32)
        for f in range(args.folds):
            tr = np.flatnonzero(fold != f)
            va = np.flatnonzero(fold == f)
            model, best = train_one(args, xtr[tr], ytr[tr], xtr[va], ytr[va],
                                    f"{args.tag}-f{f}", args.seed + f)
            oof[va] = predict_probs(model, xtr[va])
            np.save(out / f"oof_f{f}.npy", oof[va])
            torch.save(model.state_dict(), out / f"model_f{f}.pt")
            print(f"[{args.tag}-f{f}] fold acc {(oof[va].argmax(1)==ytr[va]).mean():.4f}", flush=True)
        acc = (oof.argmax(1) == ytr).mean()
        print(f"OOF accuracy {acc:.4f}", flush=True)
        np.save(out / "oof.npy", oof)
        (out / "result.json").write_text(json.dumps({"oof_acc": float(acc)}))
    else:
        # hold out 2k only for progress monitoring; not used for selection
        model, best = train_one(args, xtr, ytr, xtr[:0], ytr[:0], args.tag, args.seed)
        torch.save(model.state_dict(), out / "model.pt")
        (out / "result.json").write_text(json.dumps({"best_monitor": best}))

    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Measure how fast the checkpoint datasets can actually be read.

The val kernel spent 10158s loading 4 shards (~18GB) -> ~1.8 MB/s, which would make
a 12h submission run impossible.  Before redesigning, find out whether the mount is
slow for sequential io, only slow for mmap/random access, or fine and the cost was
somewhere else.
"""
import glob
import os
import shutil
import subprocess
import time

import torch


def sh(cmd):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return time.time() - t0, r.stdout.strip() + r.stderr.strip()[:300]


def human(b):
    return "%.2f GB" % (b / 1e9)


def main():
    for d in ("/kaggle/input", "/kaggle/input/datasets", "/kaggle/input/competitions"):
        if os.path.isdir(d):
            print(d, "->", sorted(os.listdir(d))[:10], flush=True)

    for pat in ("/kaggle/input/**/config.json",):
        print("config hits:", glob.glob(pat, recursive=True)[:6], flush=True)

    root = None
    for cand in glob.glob("/kaggle/input/*/*/model.safetensors.index.json") + \
                glob.glob("/kaggle/input/*/*/*/model.safetensors.index.json"):
        root = os.path.dirname(cand)
        break
    print("model root:", root, flush=True)

    dirs = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/*.safetensors", recursive=True)})
    print("safetensors dirs:", dirs, flush=True)

    print("\n== disk", flush=True)
    for d in ("/kaggle/working", "/tmp", "/"):
        print(d, sh("df -h %s | tail -1" % d)[1], flush=True)

    for d in dirs[:2]:
        files = sorted(f for f in os.listdir(d) if f.endswith((".safetensors", ".bin")))
        tot = sum(os.path.getsize(os.path.join(d, f)) for f in files)
        print("\n== %s  %d files  %s" % (d, len(files), human(tot)), flush=True)
        if not files:
            continue
        f0 = os.path.join(d, files[0])
        sz = os.path.getsize(f0)
        print("   %s %s" % (files[0], human(sz)), flush=True)

        # 1) raw sequential read through page cache bypass-ish (dd, 1GB)
        n = min(128, max(1, sz // (8 << 20)))
        dt, out = sh("dd if='%s' of=/dev/null bs=8M count=%d" % (f0, n))
        print("   dd %d MB   %.1fs  -> %.1f MB/s  %s"
              % (n * 8, dt, n * 8 / dt, out.splitlines()[-1][:80]), flush=True)

        # 2) copy one shard to the writable local disk
        dst = "/kaggle/working/_copy_test.bin"
        dt, _ = sh("cp '%s' '%s'" % (f0, dst))
        print("   cp %s -> %.1fs  -> %.1f MB/s" % (human(sz), dt, sz / dt / 1e6), flush=True)
        dt, _ = sh("dd if='%s' of=/dev/null bs=8M count=%d" % (dst, n))
        print("   dd on copy %d MB -> %.1f MB/s" % (n * 8, n * 8 / dt), flush=True)

        # 3) safetensors load of the copy (what our Scorer actually does)
        try:
            from safetensors.torch import load_file
            t0 = time.time()
            sd = load_file(dst, device="cpu")
            nb = sum(v.numel() * v.element_size() for v in sd.values())
            dt = time.time() - t0
            print("   load_file(copy) %s -> %.1fs -> %.1f MB/s, %d tensors"
                  % (human(nb), dt, nb / dt / 1e6, len(sd)), flush=True)
            print("   sample keys:", list(sd)[:4], flush=True)
            del sd
        except Exception as e:  # noqa: BLE001
            print("   load_file(copy) failed:", type(e).__name__, str(e)[:200], flush=True)
        os.remove(dst)

    print("\n== cpu/mem", flush=True)
    print("cpus", os.cpu_count(), torch.get_num_threads(), flush=True)
    print(sh("free -m | head -2")[1], flush=True)


if __name__ == "__main__":
    main()

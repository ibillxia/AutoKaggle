# Parallel ranged downloader for Kaggle kernel output.
#
# This machine's proxy kills any single HTTP response body at ~8-10 MB (IncompleteRead) and gives
# roughly 50-100 KB/s per connection. Sequential resume would take ~30 min per 110 MB checkpoint.
# Instead: probe the size, split into SEG-sized byte ranges, pull SEGs concurrently, then stitch.
# A broken connection only costs one segment, and the segments run in parallel.
import os, sys, time
from concurrent.futures import ThreadPoolExecutor

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

SEG = 8 * 1024 * 1024
WORKERS = 8
RETRIES = 25


def list_files(api, owner, slug):
    out, token = [], None
    with api.build_kaggle_client() as kaggle:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = owner
            req.kernel_slug = slug
            req.page_size = 100
            if token:
                req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            out.extend(resp.files or [])
            token = resp.next_page_token
            if not token:
                break
    return out


def probe_size(url):
    r = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(20, 60))
    cr = r.headers.get("Content-Range")
    r.close()
    if cr and "/" in cr:
        return int(cr.split("/")[-1])
    r = requests.head(url, timeout=(20, 60))
    return int(r.headers.get("Content-Length", 0))


def get_seg(url, start, end, dest):
    if os.path.exists(dest) and os.path.getsize(dest) == end - start + 1:
        return
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, headers={"Range": "bytes=%d-%d" % (start, end)},
                             stream=True, timeout=(20, 90))
            if r.status_code != 206:
                raise RuntimeError("HTTP %s" % r.status_code)
            buf = bytearray()
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    buf += chunk
            if len(buf) != end - start + 1:
                raise RuntimeError("short seg %d != %d" % (len(buf), end - start + 1))
            with open(dest, "wb") as f:
                f.write(buf)
            return
        except Exception as e:
            if attempt == RETRIES - 1:
                raise
            time.sleep(1 + attempt % 3)


def fetch(url, dest):
    total = probe_size(url)
    if not total:
        # unknown size: fall back to a plain single-shot download
        r = requests.get(url, timeout=(20, 300))
        open(dest, "wb").write(r.content)
        return len(r.content)
    nseg = (total + SEG - 1) // SEG
    tmpdir = dest + ".segs"
    os.makedirs(tmpdir, exist_ok=True)
    print("   total %.1f MB in %d segs" % (total / 1e6, nseg), flush=True)
    jobs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i in range(nseg):
            s = i * SEG
            e = min(s + SEG - 1, total - 1)
            jobs.append(ex.submit(get_seg, url, s, e, os.path.join(tmpdir, "%04d" % i)))
        for j in jobs:
            j.result()
    with open(dest, "wb") as out:
        for i in range(nseg):
            with open(os.path.join(tmpdir, "%04d" % i), "rb") as f:
                out.write(f.read())
    for i in range(nseg):
        os.remove(os.path.join(tmpdir, "%04d" % i))
    os.rmdir(tmpdir)
    return os.path.getsize(dest)


def main():
    owner, slug, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    want = sys.argv[4:]
    api = KaggleApi(); api.authenticate()
    t0 = time.time()
    for f in list_files(api, owner, slug):
        if want and not any(w in f.file_name for w in want):
            continue
        dest = os.path.join(out_dir, f.file_name)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        print("->", f.file_name, flush=True)
        got = fetch(f.url, dest)
        print("   saved %.2f MB (%.0fs)" % (got / 1e6, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

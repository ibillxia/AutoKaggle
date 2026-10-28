# Robust downloader for Kaggle kernel output files.
#
# The plain kernels_output() path dies at ~10 MB with IncompleteRead on this machine (proxy/MTU),
# and it has no resume: every retry restarts from zero. This script asks the API for the signed
# per-file URLs and pulls them with HTTP Range requests, appending to a .part file, so a broken
# connection costs at most one chunk instead of the whole file.
import os, sys, time, json

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

CHUNK = 4 * 1024 * 1024
RETRIES = 40


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


def fetch(url, dest):
    part = dest + ".part"
    pos = os.path.getsize(part) if os.path.exists(part) else 0
    sess = requests.Session()
    fails = 0
    while True:
        headers = {"Range": "bytes=%d-" % pos} if pos else {}
        try:
            r = sess.get(url, headers=headers, stream=True, timeout=(20, 120))
            if r.status_code not in (200, 206):
                raise RuntimeError("HTTP %s" % r.status_code)
            # 200 means the server ignored Range -> restart from zero
            if pos and r.status_code == 200:
                pos = 0
                open(part, "wb").close()
            with open(part, "ab" if pos else "wb") as f:
                for chunk in r.iter_content(CHUNK):
                    if chunk:
                        f.write(chunk)
                        pos += len(chunk)
            total = int(r.headers.get("Content-Length", 0)) + (0 if r.status_code == 200 else 0)
            break
        except Exception as e:
            fails += 1
            if fails > RETRIES:
                raise SystemExit("giving up on %s after %d tries: %s" % (os.path.basename(dest), fails, e))
            print("  retry %d at %.1f MB: %s" % (fails, pos / 1e6, str(e)[:90]), flush=True)
            time.sleep(3)
    got = os.path.getsize(part)
    os.replace(part, dest)
    return got


def main():
    owner, slug, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    want = sys.argv[4:] or None
    api = KaggleApi(); api.authenticate()
    files = list_files(api, owner, slug)
    print("remote files:", len(files))
    for f in files:
        name = f.file_name
        if want and not any(w in name for w in want):
            continue
        dest = os.path.join(out_dir, name)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        print("->", name, "size", getattr(f, "total_bytes", "?"), flush=True)
        got = fetch(f.url, dest)
        print("   saved %.2f MB" % (got / 1e6), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Kaggle 提交包装器（绕开 www.googleapis.com 不可达的问题）。

背景
----
Kaggle 普通上传赛的提交流程：
  1. POST api.kaggle.com/v1/competitions.CompetitionApiService/StartSubmissionUpload
     -> 返回 { create_url, token }
  2. PUT  <create_url>  (Google Cloud Storage resumable upload session)
     create_url 的 host 是 www.googleapis.com  -> 本机被墙 / 沙箱代理 502
  3. POST .../CreateSubmission  { blob_file_tokens: token }

解决办法
--------
www.googleapis.com 与 storage.googleapis.com 是同一套 GCS JSON API 的不同 host，
upload_id 在 query string 里，不绑定 host。把 host 换成 storage.googleapis.com 即可。

用法
----
  python validation/kaggle_submit.py --file submissions/xxx.csv \
      --competition nlp-getting-started --message "kev-0.8b ft"
  # 只试通路径、不真正创建提交
  python validation/kaggle_submit.py ... --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiCreateSubmissionRequest,
    ApiStartSubmissionUploadRequest,
)

GCS_HOST_OLD = "www.googleapis.com"
GCS_HOST_NEW = "storage.googleapis.com"


def candidate_plans(url: str) -> list[tuple[str, bool]]:
    """(url, trust_env) 候选，按成功率/速度排序。

    实测（本机）：storage.googleapis.com 直连可用；www.googleapis.com 直连超时、
    走沙箱代理 502。所以「改写 host + 直连」放第一个，「原 host + 代理」次之。
    """
    rewritten = url.replace(GCS_HOST_OLD, GCS_HOST_NEW) if GCS_HOST_OLD in url else url
    plans = [(rewritten, False), (rewritten, True), (url, True), (url, False)]
    seen, out = set(), []
    for item in plans:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def put_blob(path: str, url: str, retries: int = 2, connect_timeout: int = 10) -> bool:
    """把文件 PUT 到 GCS resumable 上传会话；按候选组合逐个尝试。"""
    size = os.path.getsize(path)
    last_err = None
    plans = candidate_plans(url)
    for attempt in range(retries):
        for target, trust_env in plans:
            host = target.split("/")[2]
            session = requests.Session()
            session.trust_env = trust_env
            try:
                with open(path, "rb") as fp:
                    resp = session.put(
                        target,
                        data=fp,
                        headers={"Content-Length": str(size)},
                        timeout=(connect_timeout, 180),
                    )
                tag = f"host={host} trust_env={trust_env}"
                if resp.status_code in (200, 201):
                    print(f"  [ok] PUT {tag} -> {resp.status_code}")
                    return True
                print(f"  [--] PUT {tag} -> {resp.status_code} {resp.text[:160]}")
                last_err = f"{resp.status_code} {resp.text[:160]}"
            except Exception as exc:  # noqa: BLE001
                print(f"  [!!] PUT host={host} trust_env={trust_env} -> {type(exc).__name__}: {str(exc)[:120]}")
                last_err = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < retries:
            time.sleep(2 * (attempt + 1))
    print(f"  上传失败，最后一次错误: {last_err}")
    return False


def submit(
    file_path: str,
    competition: str,
    message: str,
    dry_run: bool = False,
    sandbox: bool = False,
) -> int:
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        print(f"文件不存在: {file_path}")
        return 2

    size = os.path.getsize(file_path)
    print(f"提交 {file_path} ({size} bytes) -> {competition}")

    api = KaggleApi()
    api.authenticate()

    with api.build_kaggle_client() as kaggle:
        svc = kaggle.competitions.competition_api_client

        req = ApiStartSubmissionUploadRequest()
        req.competition_name = competition
        req.file_name = os.path.basename(file_path)
        req.content_length = size
        req.last_modified_epoch_seconds = int(os.path.getmtime(file_path))

        resp = svc.start_submission_upload(req)
        create_url = resp.create_url
        token = resp.token
        print(f"  create_url = {create_url}")
        print(f"  token      = {token}")

        ok = put_blob(file_path, create_url)
        if not ok:
            return 3

        if dry_run:
            print("dry-run：上传通路 OK，跳过 CreateSubmission")
            return 0

        sub = ApiCreateSubmissionRequest()
        sub.competition_name = competition
        sub.blob_file_tokens = token
        if message:
            sub.submission_description = message
        if sandbox:
            sub.sandbox = True

        result = svc.create_submission(sub)
        print("CreateSubmission ->", result)
        return 0


def submit_code(competition: str, kernel: str, version: int, file_name: str, message: str) -> int:
    """Code competitions: no file is uploaded, the API just points at a finished kernel version.

    Nothing here touches GCS, so this path has none of the host problems `submit` works around.
    """
    api = KaggleApi()
    api.authenticate()
    try:
        result = api.competition_submit_code(file_name, message, competition, kernel, version, quiet=False)
    except TypeError:                                    # older signature
        result = api.competition_submit_code(file_name, message, competition, kernel, version)
    print("CreateCodeSubmission ->", result)
    try:
        limits = api.competition_get_submission_limits(competition)
        print("submissions remaining today:", limits.num_allowed_now)
    except Exception as exc:  # noqa: BLE001
        print("limits unavailable:", type(exc).__name__, exc)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="待提交的 CSV，或代码赛的内核输出文件名")
    ap.add_argument("--competition", required=True)
    ap.add_argument("--message", default="")
    ap.add_argument("--dry-run", action="store_true", help="只验证上传通路，不创建提交")
    ap.add_argument("--sandbox", action="store_true", help="标记为 sandbox 提交")
    ap.add_argument("--kernel", default="", help="代码赛：<owner>/<notebook>")
    ap.add_argument("--version", type=int, default=0, help="代码赛：内核版本号")
    args = ap.parse_args()
    if args.kernel:
        if not args.version:
            ap.error("--version is required with --kernel")
        return submit_code(args.competition, args.kernel, args.version, args.file, args.message)
    return submit(args.file, args.competition, args.message, args.dry_run, args.sandbox)


if __name__ == "__main__":
    sys.exit(main())

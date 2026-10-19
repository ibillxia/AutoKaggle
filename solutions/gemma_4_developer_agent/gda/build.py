"""Assemble a candidate's file tree into a real ``submission.zip``."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict

from gda import paths


def stage_candidate(name: str, files: Dict[str, str], dest_root: Path | None = None) -> Path:
    """Write the candidate's files to disk under ``dest_root/<name>/``."""
    dest_root = dest_root or paths.WORK_DIR
    out = dest_root / name
    for rel, content in files.items():
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
    return out


def build_zip(name: str, files: Dict[str, str],
              zip_path: Path | None = None) -> Path:
    """Write ``submission.zip`` with deterministic metadata.

    The entry order is sorted and timestamps are fixed so two builds of the same
    candidate produce identical bytes -- which is what lets the comparison be
    reproducible.
    """
    paths.ensure_dirs()
    zip_path = zip_path or (paths.SUBMISSIONS_DIR / name / "submission.zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(files):
            info = zipfile.ZipInfo(filename=rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, files[rel].encode("utf-8"))
    return zip_path


def read_zip(zip_path: Path) -> Dict[str, str]:
    """Read a submission zip back into {relative path: text}."""
    out: Dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            out[info.filename] = zf.read(info).decode("utf-8", "replace")
    return out


def build_all() -> Dict[str, Path]:
    from gda.candidates import CANDIDATES, build_candidate

    paths.ensure_dirs()
    built = {}
    for name in CANDIDATES:
        files = build_candidate(name)
        stage_candidate(name, files)
        built[name] = build_zip(name, files)
    return built

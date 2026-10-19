"""Build, validate, score and rank every candidate, then select the best.

Selection rule (declared once, here, so it can be argued with):

    composite = compliance x ( 0.60 x rubric_norm + 0.40 x proxy_norm )

* ``compliance`` is 0/1: any failing HARD gate disqualifies the candidate, because
  the harness would reject the upload outright.
* ``rubric_norm`` = rubric score / 10.
* ``proxy_norm`` = fraction of executable proxy checks passed, or **0.5** when the
  candidate ships nothing executable to check. 0.5 is a deliberately neutral
  prior: a prompt-only agent is neither penalised as broken nor credited with
  engineering it cannot demonstrate.

The rubric leads because it is the only signal available for every candidate;
the proxy is strong evidence but only exists for candidates that ship scripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from gda import build, paths
from gda.candidates import CANDIDATES, build_candidate
from gda.proxy import build_fixtures, run_proxy  # noqa: F401  (build_fixtures re-exported)
from gda.rubric import score_candidate
from gda.validate import hard_failures, soft_failures, validate

RUBRIC_WEIGHT = 0.60
PROXY_WEIGHT = 0.40
UNVERIFIED_PRIOR = 0.50


@dataclass
class CandidateResult:
    name: str
    idea: str
    delta: str
    zip_path: Optional[Path]
    n_files: int
    checks: List[Dict] = field(default_factory=list)
    rubric: Dict = field(default_factory=dict)
    proxy: Dict = field(default_factory=dict)

    @property
    def hard_failures(self) -> List[Dict]:
        return hard_failures(self.checks)

    @property
    def soft_failures(self) -> List[Dict]:
        return soft_failures(self.checks)

    @property
    def compliant(self) -> bool:
        return not self.hard_failures

    @property
    def rubric_norm(self) -> float:
        return self.rubric.get("score_10", 0.0) / 10.0

    @property
    def proxy_norm(self) -> float:
        score = self.proxy.get("score")
        return UNVERIFIED_PRIOR if score is None else float(score)

    @property
    def composite(self) -> float:
        if not self.compliant:
            return 0.0
        return RUBRIC_WEIGHT * self.rubric_norm + PROXY_WEIGHT * self.proxy_norm

    def as_dict(self) -> Dict:
        return {
            "name": self.name,
            "idea": self.idea,
            "delta": self.delta,
            "zip": str(self.zip_path) if self.zip_path else None,
            "n_files": self.n_files,
            "compliant": self.compliant,
            "hard_failures": [c["id"] for c in self.hard_failures],
            "soft_warnings": [c["id"] for c in self.soft_failures],
            "rubric": self.rubric,
            "proxy": self.proxy,
            "rubric_norm": round(self.rubric_norm, 4),
            "proxy_norm": round(self.proxy_norm, 4),
            "composite": round(self.composite, 4),
            "checks": self.checks,
        }


def evaluate(work_dir: Path | None = None, run_proxy_checks: bool = True) -> Dict:
    paths.ensure_dirs()
    work_dir = work_dir or paths.WORK_DIR
    fixtures = build_fixtures(work_dir / "fixtures")

    results: List[CandidateResult] = []
    for name, meta in CANDIDATES.items():
        files = build_candidate(name)
        staged = build.stage_candidate(name, files, dest_root=work_dir)
        zip_path = build.build_zip(name, files)

        checks = validate(files)
        rubric = score_candidate(files).as_dict()
        if run_proxy_checks:
            proxy = run_proxy(name, files, staged, fixtures).as_dict()
        else:
            proxy = {"candidate": name, "executed": 0, "passed": 0,
                     "score": None, "checks": []}

        results.append(CandidateResult(
            name=name, idea=meta["idea"], delta=meta["delta"],
            zip_path=zip_path, n_files=len(files),
            checks=checks, rubric=rubric, proxy=proxy,
        ))

    ranked = sorted(results, key=lambda r: r.composite, reverse=True)
    best = ranked[0]

    # publish the winner as artifacts/best_submission.zip
    if best.zip_path and best.compliant:
        import shutil
        shutil.copy2(best.zip_path, paths.BEST_ZIP)

    payload = {
        "ranking_rule": {
            "composite": "compliance x (0.60 x rubric_norm + 0.40 x proxy_norm)",
            "unverified_prior": UNVERIFIED_PRIOR,
            "note": "compliance is 0/1 over hard harness gates",
        },
        "best": best.name,
        "best_zip": str(paths.BEST_ZIP) if best.compliant else None,
        "order": [r.name for r in ranked],
        "candidates": [r.as_dict() for r in ranked],
    }

    out = paths.EVALUATION_DIR / "comparison.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    _write_csv(ranked, paths.EVALUATION_DIR / "comparison.csv")
    return payload


def _write_csv(ranked: List[CandidateResult], path: Path) -> None:
    import csv

    dims = ["budget_discipline", "navigation_protocol", "verification_loop",
            "patch_hygiene", "context_compaction", "prompt_clarity", "skill_reuse"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "candidate", "compliant", "hard_gate_failures",
                    "rubric_10", *[f"rub_{d}" for d in dims],
                    "proxy_executed", "proxy_passed", "proxy_score",
                    "rubric_norm", "proxy_norm", "composite", "n_files"])
        for i, r in enumerate(ranked, 1):
            d = r.rubric.get("dimensions", {})
            w.writerow([
                i, r.name, int(r.compliant),
                "|".join(c["id"] for c in r.hard_failures) or "",
                round(r.rubric.get("score_10", 0.0), 3),
                *[round(d.get(x, {}).get("score", 0.0), 3) for x in dims],
                r.proxy.get("executed"), r.proxy.get("passed"),
                "" if r.proxy.get("score") is None else round(r.proxy["score"], 4),
                round(r.rubric_norm, 4), round(r.proxy_norm, 4),
                round(r.composite, 4), r.n_files,
            ])


def load_comparison() -> Dict:
    path = paths.EVALUATION_DIR / "comparison.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m gda.cli evaluate` first")
    return json.loads(path.read_text(encoding="utf-8"))

"""Comparison figures (light theme, deterministic).

matplotlib is optional: if it is missing the report still generates, and the
figure section records that figures were skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from gda import paths
from gda.rubric import WEIGHTS

INK = "#1f2933"
MUTED = "#7b8794"
OK = "#2f855a"
BAD = "#c53030"
ACCENT = "#2b6cb0"
GRID = "#e1e5ea"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
    })
    return plt


def _finish(fig, path: Path, plt):
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def figure_composite(payload: Dict, out_dir: Path) -> Path | None:
    try:
        plt = _plt()
    except Exception:
        return None
    cands = payload["candidates"]
    names = [c["name"] for c in cands][::-1]
    vals = [c["composite"] for c in cands][::-1]
    colors = [OK if c["compliant"] else BAD for c in cands][::-1]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bars = ax.barh(names, vals, color=colors, height=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, max(vals + [0.1]) * 1.18)
    ax.set_xlabel("composite score (compliance x [0.60 rubric + 0.40 proxy])")
    ax.set_title(f"Gemma 4 Developer Agent — candidate ranking\n"
                 f"best: {payload['best']}", loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    p = out_dir / "fig01_composite_ranking.png"
    _finish(fig, p, plt)
    return p


def figure_rubric(payload: Dict, out_dir: Path) -> Path | None:
    try:
        plt = _plt()
    except Exception:
        return None
    cands = payload["candidates"]
    dims = list(WEIGHTS)
    names = [c["name"] for c in cands]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bottoms = [0.0] * len(names)
    cmap = ["#2b6cb0", "#3182ce", "#63b3ed", "#90cdf4", "#bee3f8",
            "#cbd5e0", "#a0aec0"]
    for i, dim in enumerate(dims):
        vals = [c["rubric"]["dimensions"][dim]["score"] * WEIGHTS[dim] * 10
                for c in cands]
        ax.bar(names, vals, bottom=bottoms, label=f"{dim} (w={WEIGHTS[dim]:.2f})",
               color=cmap[i % len(cmap)], edgecolor="white", linewidth=0.6)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for x, total in enumerate(bottoms):
        ax.text(x, total + 0.12, f"{total:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_ylabel("rubric contribution (0-10)")
    ax.set_title("Rubric decomposition — which engineering idea earns the points",
                 loc="left", fontsize=10)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, -0.18))
    ax.grid(axis="x", visible=False)
    p = out_dir / "fig02_rubric_decomposition.png"
    _finish(fig, p, plt)
    return p


def figure_proxy(payload: Dict, out_dir: Path) -> Path | None:
    try:
        plt = _plt()
    except Exception:
        return None
    cands = payload["candidates"]
    names = [c["name"] for c in cands]
    ex = [c["proxy"]["executed"] for c in cands]
    pa = [c["proxy"]["passed"] for c in cands]

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = range(len(names))
    ax.bar(x, ex, color="#cbd5e0", width=0.6, label="executed checks")
    ax.bar(x, pa, color=OK, width=0.6, label="passed")
    for i, (e, p_) in enumerate(zip(ex, pa)):
        ax.text(i, e + 0.15, f"{p_}/{e}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("proxy checks")
    ax.set_title("Proxy — scripts actually executed against fixture repos",
                 loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="x", visible=False)
    p = out_dir / "fig03_proxy_execution.png"
    _finish(fig, p, plt)
    return p


def figure_gates(payload: Dict, out_dir: Path) -> Path | None:
    try:
        plt = _plt()
    except Exception:
        return None
    cands = payload["candidates"]
    gate_ids: List[str] = []
    for c in cands:
        for chk in c["checks"]:
            if chk["id"] not in gate_ids:
                gate_ids.append(chk["id"])

    matrix = []
    for c in cands:
        by_id = {chk["id"]: chk for chk in c["checks"]}
        row = []
        for g in gate_ids:
            chk = by_id.get(g)
            row.append(2 if chk is None else (1 if chk["passed"] else 0))
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(gate_ids)))
    ax.set_xticklabels(gate_ids, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(cands)))
    ax.set_yticklabels([c["name"] for c in cands], fontsize=8)
    ax.set_title("Harness gates — green pass / red fail / yellow not applicable",
                 loc="left", fontsize=10)
    ax.grid(False)
    p = out_dir / "fig04_gate_matrix.png"
    _finish(fig, p, plt)
    return p


def make_all(payload: Dict) -> List[Path]:
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (figure_composite, figure_rubric, figure_proxy, figure_gates):
        p = fn(payload, paths.FIGURES_DIR)
        if p:
            made.append(p)
    return made

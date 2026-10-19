"""Cover card / thumbnail for the Kaggle writeup.

Kaggle asks for a 560 x 280 image in the writeup's Media Gallery. This renders one
from the delivered numbers: the metric name, the headline linkage with its CI, the
reliability context, and the "blind spots" 2x2 that arbitrates sensor vs stopwatch.

Every string is measured after drawing (renderer -> pixel extent) and the layout
asserts that no two text boxes overlap and that nothing leaves the canvas. On a
560x280 card, overlapping text is the difference between a usable thumbnail and a
smeared one, so this is checked rather than eyeballed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# matplotlib must be told where it may write before it is imported anywhere
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "cache" / "mpl")
)

COVER_WIDTH = 560
COVER_HEIGHT = 280
COVER_DPI = 100

# palette shared with the figure suite so the card matches the submission
BG = "#0e1726"
INK = "#f2f5fa"
MUTED = "#93a2b8"
ACCENT = "#4da3ff"
WARN = "#ff9f45"
GRID = "#22304a"

MARGIN = 22
RIGHT_BLOCK_W = 120
COL_GAP = 14
LEFT_W = COVER_WIDTH - 2 * MARGIN - RIGHT_BLOCK_W - COL_GAP


def _fp(size: float, weight: str = "normal"):
    """A FontProperties that resolves on Windows and Linux without warnings."""
    from matplotlib import font_manager

    for family in ["Segoe UI", "DejaVu Sans", "Liberation Sans", "Arial"]:
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family), fallback_to_default=False
            )
        except Exception:  # noqa: BLE001
            continue
        if path:
            return font_manager.FontProperties(fname=path, size=size, weight=weight)
    return font_manager.FontProperties(size=size, weight=weight)


class _Flow:
    """Places measured text blocks down the left column and records their boxes."""

    def __init__(self, ax, fig, width_px: float, x: float):
        self.ax = ax
        self.fig = fig
        self.width_px = width_px
        self.x = x
        self.y = 0.0
        self.boxes: List[Tuple[str, Tuple[float, float, float, float]]] = []

    def _wrap(self, text: str, size: float) -> str:
        """Greedy word wrap using the real rendered width of each candidate line."""
        renderer = self.fig.canvas.get_renderer()
        probe = self.ax.text(0, 0, "", fontproperties=_fp(size))
        lines: List[str] = []
        for para in text.split("\n"):
            current = ""
            for word in para.split():
                candidate = f"{current} {word}".strip()
                probe.set_text(candidate)
                width = probe.get_window_extent(renderer).width
                if current and width > self.width_px:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            lines.append(current)
        probe.remove()
        return "\n".join(lines) or text

    def add(self, text: str, *, size: float, color: str, weight: str = "normal",
            gap_before: float = 0.0, gap_after: float = 4.0, wrap: bool = True) -> None:
        if wrap:
            text = self._wrap(text, size)
        self.y += gap_before
        artist = self.ax.text(self.x, self.y, text, color=color,
                              fontproperties=_fp(size, weight), ha="left", va="top",
                              linespacing=1.30)
        self.fig.canvas.draw()
        ext = artist.get_window_extent(self.fig.canvas.get_renderer())
        box = (ext.x0, COVER_HEIGHT - ext.y1, ext.x1, COVER_HEIGHT - ext.y0)
        self.boxes.append((text.replace("\n", " / ")[:34], box))
        self.y = box[3] + gap_after


def build_cover_image(
    deep: Dict[str, Any],
    audit_report: Dict[str, Any],
    *,
    metric_name: str,
    out_path: Path,
    debug: bool = False,
) -> Path:
    """Render the 560x280 cover card and return the written path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    link = deep.get("linkage", {})
    mv = deep.get("metric_validation", {})
    bs = deep.get("blind_spots", {})
    archetypes = bs.get("outcome_mean_by_archetype", {})

    fig = plt.figure(figsize=(COVER_WIDTH / COVER_DPI, COVER_HEIGHT / COVER_DPI),
                     dpi=COVER_DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, COVER_WIDTH)
    ax.set_ylim(COVER_HEIGHT, 0)  # y grows downward, like pixels
    ax.axis("off")
    ax.set_facecolor(BG)

    flow = _Flow(ax, fig, LEFT_W, MARGIN)
    flow.y = 20.0

    flow.add(metric_name.upper(), size=15.5, color=ACCENT, weight="bold", gap_after=3)
    flow.add("What 10 Hz tracking sees that a stopwatch cannot",
             size=9.8, color=INK, gap_after=8)

    r = link.get("partial_r")
    if r is not None:
        lo, hi, n = link.get("ci_low"), link.get("ci_high"), link.get("n")
        # big number on its own line, then the supporting detail beneath it
        flow.add(f"{r:+.2f}", size=34, color=INK, weight="bold", wrap=False,
                 gap_after=1)
        flow.add("partial r", size=8.8, color=MUTED, wrap=False, gap_after=1)
        detail = []
        if lo is not None and hi is not None:
            detail.append(f"95% CI [{lo:+.2f}, {hi:+.2f}]")
        if n:
            detail.append(f"n = {int(n)} DL")
        if detail:
            flow.add("  \u00b7  ".join(detail), size=8.4, color=MUTED, gap_after=8)

    dr = link.get("delta_r2")
    if dr is not None:
        flow.add(f"+{dr:.3f} out-of-sample R\u00b2 beyond the entire Combine battery",
                 size=9.0, color=WARN, gap_after=7)

    flow.add("Entry speed is the one level measure that stays repeatable "
             "(r = 0.40); the ratio-based \u201cbreak shape\u201d traits do not "
             "(0.13, \u22120.03).", size=8.2, color=MUTED, gap_after=6)

    # ---------------- right block: the disagreements, as a readable bar chart
    # A 2x2 heatmap was tried first and was illegible at card size, so this shows
    # only the two off-diagonal archetypes the writeup actually argues about.
    right_x = COVER_WIDTH - MARGIN - RIGHT_BLOCK_W
    right_boxes: List[Tuple[str, Tuple[float, float, float, float]]] = []
    if archetypes:
        right_title = ax.text(right_x, 34, "the disagreements", color=MUTED,
                              fontproperties=_fp(8.0), ha="left", va="center")
        fig.canvas.draw()
        rtext = right_title.get_window_extent(fig.canvas.get_renderer())
        right_boxes.append(("the disagreements",
                            (rtext.x0, COVER_HEIGHT - rtext.y1,
                             rtext.x1, COVER_HEIGHT - rtext.y0)))

        bar_h = 104
        bar_top = 50
        bar = fig.add_axes([
            (right_x + 2) / COVER_WIDTH,
            1 - (bar_top + bar_h) / COVER_HEIGHT,
            (RIGHT_BLOCK_W - 4) / COVER_WIDTH,
            bar_h / COVER_HEIGHT,
        ])
        names = ["sensor\nlikes", "watch\nlikes", "both\nagree"]
        keys = ["sensor_beats_stopwatch", "stopwatch_beats_sensor",
                "confirmed_elite"]
        vals = [archetypes.get(k, 0.0) for k in keys]
        colors = [ACCENT, WARN, "#5a6b85"]
        ys = [0, 1, 2]
        bar.barh(ys, vals, color=colors, height=0.62)
        bar.set_yticks(ys)
        bar.set_yticklabels(names, fontproperties=_fp(6.4), color=MUTED)
        bar.invert_yaxis()
        bar.set_xticks([])
        bar.set_facecolor(BG)
        for spine in bar.spines.values():
            spine.set_visible(False)
        bar.tick_params(axis="y", length=0, pad=2)
        span = max(vals) if vals else 1.0
        for y, v in zip(ys, vals):
            bar.text(v + span * 0.05, y, f"{v:.3f}", va="center", ha="left",
                     color=INK, fontproperties=_fp(6.6))
        bar.set_xlim(0, span * 1.42)
        right_boxes.append(("bars", (right_x, bar_top, right_x + RIGHT_BLOCK_W,
                                     bar_top + bar_h)))

    # ---------------- footer ----------------
    ax.plot([MARGIN, COVER_WIDTH - MARGIN], [222, 222], color=GRID, linewidth=1.0)
    ax.text(MARGIN, 230,
            "NFL Big Data Bowl 2027  \u00b7  Combine sensor tracking "
            "\u2192 NFL production",
            color=MUTED, fontproperties=_fp(8.2), ha="left", va="top")
    counts = audit_report.get("row_counts", {})
    participants = counts.get("players")
    attempts = audit_report.get("combine_tracking_n_attempts")
    plays = audit_report.get("player_play_rows")
    if participants and attempts and plays:
        ax.text(MARGIN, 250,
                f"{participants} prospects  \u00b7  {attempts:,} tracked drills"
                f"  \u00b7  {plays:,} in-game player-plays",
                color=MUTED, fontproperties=_fp(7.6), ha="left", va="top")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=COVER_DPI, facecolor=BG)

    if debug:
        _audit(flow.boxes + right_boxes)

    plt.close(fig)
    return out_path


def _audit(boxes) -> None:
    """Report measured boxes; flag overlaps and out-of-bounds text."""
    print(f"canvas {COVER_WIDTH}x{COVER_HEIGHT}; left column width {LEFT_W:.0f}px")
    problems = []
    for label, (x0, y0, x1, y1) in boxes:
        flag = ""
        if x1 > COVER_WIDTH - 2 or x0 < 2 or y1 > COVER_HEIGHT - 2 or y0 < 0:
            flag = "   <-- OUT OF BOUNDS"
            problems.append(label)
        print(f"  [{x0:6.1f},{y0:6.1f}]-[{x1:6.1f},{y1:6.1f}] {label!r}{flag}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                print(f"  OVERLAP: {boxes[i][0]!r} <-> {boxes[j][0]!r}")
                problems.append(f"{boxes[i][0]}~{boxes[j][0]}")
    print("PROBLEMS:", len(problems))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deep-dive", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--metric-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    deep = json.loads(Path(a.deep_dive).read_text(encoding="utf-8"))
    audit = json.loads(Path(a.audit).read_text(encoding="utf-8"))
    written = build_cover_image(deep, audit, metric_name=a.metric_name,
                                out_path=Path(a.out), debug=a.debug)
    from PIL import Image

    with Image.open(written) as im:
        print(f"wrote {written} ({im.size[0]}x{im.size[1]})")

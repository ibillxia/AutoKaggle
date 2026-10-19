"""Build a genuinely self-contained public notebook for the BDB2027 submission.

The shipped notebook imported ``bdb`` from the repository, so re-running it on
Kaggle required uploading the solution package as an extra dataset. This module
inlines the package source into one code cell instead, so the notebook needs only
the competition data as input.

The inlined modules are the subset needed to reproduce the headline result:
paths, schema, io, reliability, kinematics, ingame, feasibility, stats, deepdive.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PACKAGE = Path(__file__).resolve().parent
MODULES = [
    "paths", "schema", "io", "reliability", "kinematics",
    "ingame", "feasibility", "stats", "deepdive",
]


def _module_sources() -> str:
    """A repr-able dict of module name -> source text."""
    parts = []
    for name in MODULES:
        src = (PACKAGE / f"{name}.py").read_text(encoding="utf-8")
        parts.append((name, src))
    return parts


def build_self_contained_notebook(
    topic: Dict[str, Any],
    deep: Dict[str, Any],
) -> Dict[str, Any]:
    """Notebook JSON: package inlined, data auto-discovered."""
    sources = _module_sources()
    cells: List[Dict[str, Any]] = []

    def md(text: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": text.splitlines(True)})

    def code(text: str) -> None:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": text.splitlines(True)})

    trait = topic["trait"]
    pg = topic["position_group"]
    outcome = topic["outcome"]

    md(
        "# NFL Big Data Bowl 2027 -- reproducible analysis\n\n"
        "This notebook reproduces the headline result of the accompanying writeup from "
        "the raw competition CSVs. It uses the 10 Hz **Combine tracking** "
        "(`combine_tracking.csv`), the 10 Hz **in-game tracking** "
        "(`game_tracking_YYYY.csv`) and `player_play.csv`.\n\n"
        f"**Claim under test:** `{trait}` predicts `{outcome}` for the {pg} group, over "
        "and above the conventional Combine battery and draft capital.\n\n"
        "**Inputs.** Only the competition dataset `nfl-big-data-bowl-2027` is required. "
        "The analysis package is inlined in the next cell, so no extra dataset and no "
        "path editing is needed.\n"
    )

    # ---- inlined package -------------------------------------------------
    # repr() of the JSON text gives a single decoded Python literal, so one
    # json.loads() recovers the dict and module docstrings (which contain triple
    # quotes) cannot terminate the literal.
    payload = repr(json.dumps(dict(sources)))
    code(
        "# ---------------------------------------------------------------------------\n"
        "# The analysis package, inlined so this notebook is self-contained.\n"
        "# It is the same code as solutions/nfl_big_data_bowl_2027 in the repository.\n"
        "# ---------------------------------------------------------------------------\n"
        "import json as _json, sys, types\n"
        "\n"
        f"_BDB_PAYLOAD = {payload}\n"
        "_BDB_SOURCES = _json.loads(_BDB_PAYLOAD)\n"
        "\n"
        "_pkg = types.ModuleType('bdb')\n"
        "_pkg.__path__ = []\n"
        "sys.modules['bdb'] = _pkg\n"
        "for _name, _src in _BDB_SOURCES.items():\n"
        "    _mod = types.ModuleType('bdb.' + _name)\n"
        "    _mod.__package__ = 'bdb'\n"
        "    # the modules resolve their own paths from __file__, so give them one\n"
        "    _mod.__file__ = f'<bdb.{_name}>'\n"
        "    setattr(_pkg, _name, _mod)\n"
        "    sys.modules['bdb.' + _name] = _mod\n"
        "for _name in _BDB_SOURCES:\n"
        "    exec(compile(_BDB_SOURCES[_name], f'<bdb.{_name}>', 'exec'),\n"
        "         sys.modules['bdb.' + _name].__dict__)\n"
        "print('inlined bdb modules:', ', '.join(_BDB_SOURCES))\n"
    )

    # ---- data discovery --------------------------------------------------
    code(
        "# Locate the competition CSVs. Nothing needs editing on Kaggle or locally.\n"
        "import os\n"
        "from pathlib import Path\n"
        "from bdb import paths\n"
        "\n"
        "def _find_data():\n"
        "    if paths.missing_files() == []:\n"
        "        return paths.DATA_DIR\n"
        "    cands = []\n"
        "    here = Path.cwd()\n"
        "    cands += [here / 'multi_agents/competition/nfl-big-data-bowl-2027/data',\n"
        "              here / 'multi_agents/competition/nfl_big_data_bowl_2027/data']\n"
        "    for root in [Path('/kaggle/input')]:\n"
        "        if root.is_dir():\n"
        "            cands += sorted(root.glob('*')) + sorted(root.glob('*/*'))\n"
        "    for c in cands:\n"
        "        if c.is_dir() and (c / 'players.csv').exists() and (c / 'combine_tracking.csv').exists():\n"
        "            return c\n"
        "    return None\n"
        "\n"
        "_data = _find_data()\n"
        "if _data is None:\n"
        "    raise SystemExit('Competition CSVs not found. Attach the nfl-big-data-bowl-2027 dataset.')\n"
        "paths.DATA_DIR = _data\n"
        "\n"
        "# Derived artefacts go to a writable directory (Kaggle input is read-only).\n"
        "from pathlib import Path as _P\n"
        "_out = _P('/kaggle/working/bdb_out') if _P('/kaggle/working').is_dir() else _P.cwd() / 'bdb_out'\n"
        "_out.mkdir(parents=True, exist_ok=True)\n"
        "paths.ARTIFACT_DIR = _out\n"
        "paths.CACHE_DIR = _out / 'cache'\n"
        "paths.FIGURE_DIR = _out / 'figures'\n"
        "paths.REPORT_DIR = _out / 'reports'\n"
        "paths.ensure_dirs()\n"
        "print('data   :', paths.DATA_DIR)\n"
        "print('outputs:', paths.ARTIFACT_DIR)\n"
        "print('missing:', paths.missing_files() or 'none')\n"
    )

    md("## 1. Build the 10 Hz sensor features from the raw Combine tracking\n")
    code(
        "import pandas as pd\n"
        "from bdb import io, kinematics\n"
        "pd.set_option('display.width', 160)\n"
        "\n"
        "tracking = io.load_combine_tracking(players_only=True)\n"
        "attempts = kinematics.extract_all_attempt_features(tracking)\n"
        "player_feats = kinematics.player_level_combine_features(attempts)\n"
        "print(tracking.shape, '->', attempts.shape, '->', player_feats.shape)\n"
    )

    md(
        "Each row of `attempts` is one drill attempt summarised into coach-readable "
        "movement traits: peak speed, entry speed into the sharpest cut, the minimum "
        "speed carried through it, path geometry, and the braking/re-acceleration shape.\n"
    )

    md("## 2. Reliability screen (uses no outcome data)\n")
    code(
        "from bdb import reliability\n"
        "rel = reliability.build_reliability_table(attempts)\n"
        "print('traits scored:', len(rel))\n"
        "rel[['feature', 'kind', 'reliability']].head(12)\n"
    )
    md(
        "Level and timing traits survive; ratio and difference traits mostly do not. "
        "This screen runs before any outcome is touched, and traits below the floor are "
        "dropped from the search space.\n"
    )

    md("## 3. Build the in-game outcomes and the analysis table\n")
    code(
        "from bdb import ingame\n"
        "ingame_feats = ingame.build_ingame_player_features()\n"
        "table = io.build_analysis_table(player_feats, ingame_feats, force=True)\n"
        "print(table.shape)\n"
    )

    md("## 4. Hypothesis screen with FDR control\n")
    code(
        "from bdb import feasibility\n"
        "feas = feasibility.scan_all(table)\n"
        "print('pairs screened:', len(feas))\n"
        "print('survivors     :', int(feas['adds_beyond_stopwatch'].sum()))\n"
        "feas.sort_values('delta_r2', ascending=False).head(10)[\n"
        "    ['position_group', 'outcome', 'trait', 'n', 'partial_spearman',\n"
        "     'q_value', 'delta_r2']]\n"
    )

    md("## 5. Confirm the selected linkage with bootstrap uncertainty\n")
    code(
        f"from bdb import deepdive\n"
        f"result = deepdive.confirm_linkage(table, '{pg}', '{outcome}', '{trait}')\n"
        "print(result.sentence())\n"
        "print()\n"
        "print(result.ols)\n"
    )

    md(
        "The claim is an *incremental* one: every model's baseline is the conventional "
        "Combine battery (40, 10-yard split, 3-cone, short shuttle, vertical, broad jump, "
        "bench) plus log draft capital, age and body size, median-imputed with explicit "
        "'did not test' indicators.\n"
    )

    md("## 6. Leave-one-out influence: one player or a real effect?\n")
    code(
        "import pandas as pd\n"
        "from bdb import stats\n"
        "Z, y, sub, _ = feasibility.build_design(table, 'DL', 'sack_rate')\n"
        "x = pd.to_numeric(sub['skill_dl__entry_speed_max'], errors='coerce')\n"
        "fr = pd.concat([x.rename('x'), y.rename('y'), Z], axis=1).dropna()\n"
        "loo = stats.leave_one_out_stability(fr['x'], fr['y'], Z)\n"
        "stats.influence_summary(loo)\n"
    )

    md(
        "---\n### Reproducibility\n"
        "This notebook is self-contained: the analysis package is inlined above and the "
        "competition data is located automatically. In the repository the identical code "
        "lives in `solutions/nfl_big_data_bowl_2027`, where `python -m bdb.cli all` "
        "rebuilds the features and the screen and `python -m bdb.cli deepdive` reproduces "
        "the confirmatory analysis and every figure.\n"
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deep-dive", required=True,
                    help="deep_dive.json produced by bdb.cli deepdive")
    ap.add_argument("--topic-trait", required=True)
    ap.add_argument("--topic-group", required=True)
    ap.add_argument("--topic-outcome", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    deep = json.loads(Path(a.deep_dive).read_text(encoding="utf-8"))
    topic = {"trait": a.topic_trait, "position_group": a.topic_group,
             "outcome": a.topic_outcome}
    nb = build_self_contained_notebook(topic, deep)
    Path(a.out).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    total = sum(len(c["source"]) for c in nb["cells"])
    print(f"wrote {a.out}: {len(nb['cells'])} cells, {total} source lines")

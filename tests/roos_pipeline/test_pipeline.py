"""Integration test: all four stages on the fake chat, in a temporary folder.

The unit tests check each step alone; this one checks that the files between the
stages fit together, and that each stage can run from the previous stage's file
alone, which is the whole point of the boundaries.
"""

from __future__ import annotations

import json

import matplotlib
import pandas as pd
import pytest

from roos_pipeline import aggregate, features, figures, prepare
from roos_pipeline.settings import Paths

matplotlib.use("Agg")


@pytest.fixture
def paths(tmp_path, aliases) -> Paths:
    """A throwaway project root with only a config.toml holding the aliases."""
    lines = ["[author_aliases]"] + [f'"{k}" = "{v}"' for k, v in aliases.items()]
    (tmp_path / "config.toml").write_text("\n".join(lines), encoding="utf-8")
    return Paths(root=tmp_path)


def test_stages_run_one_after_another(paths, fake_chat):
    prepare.run(paths, raw=fake_chat)
    features.run(paths)
    aggregate.run(paths, n_shuffle=200)

    weekly = pd.read_parquet(paths.weekly_symptoms)
    hard = weekly[weekly["group"] == aggregate.HARD]
    normal = weekly[weekly["group"] == aggregate.NORMAL]
    # The fake chat has exactly 3 tired messages per hard week, 1 per normal week.
    assert len(hard) == 10
    assert (hard["either"] == 3).all()
    assert (normal["either"] == 1).all()
    assert 4 not in weekly.loc[weekly["group"] != "other", "pregnancy_week"].to_list()

    stats = json.loads(paths.symptom_stats.read_text())
    assert stats["gap_observed"] == pytest.approx(2.0)

    # The language shares of each period add up to 100%.
    lang = pd.read_parquet(paths.language_share)
    assert lang.groupby("move_period")["pct"].sum().to_numpy() == pytest.approx(100)


def test_figures_stage_needs_only_the_stage_3_files(paths, fake_chat):
    prepare.run(paths, raw=fake_chat)
    features.run(paths)
    aggregate.run(paths, n_shuffle=200)
    # Delete stages 1 and 2's output: redrawing must not need them.
    paths.messages.unlink()
    paths.features.unlink()

    # The fake chat is 3x by construction, so the "three times" guard passes.
    figures.run(paths)
    assert len(list(paths.figures.glob("*.png"))) == 5


def test_three_times_guard_stops_a_stale_claim():
    stats = {"mean_hard": 10.0, "mean_normal": 5.0}  # 2x, not 3x
    with pytest.raises(ValueError, match="three times"):
        figures.prepare_and_plan(stats)

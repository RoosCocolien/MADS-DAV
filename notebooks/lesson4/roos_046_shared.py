"""Shared setup for the roos_04.6a-d notebooks: loading, dates, saving, and the helpers
§4.6.13 reuses from §4.6.12.

The dates, `normalise`, `SYMPTOM_PATTERNS` and `fit_daily` now come from `roos_pipeline`
(2026-10-06), so the notebooks and the pipeline cannot drift apart. roos_04.6c still
shows its own copy of the patterns and the fit, to explain them (§4.6.12 steps 1 and 3).
"""

import re
from pathlib import Path

import pandas as pd
from goad_toolkit.visualizer import PlotSettings

from roos_pipeline import settings
from roos_pipeline.aggregate import aic, fit_daily, nbinom_from  # noqa: F401 (re-exported)
from roos_pipeline.features import normalise  # noqa: F401 (re-exported)
from wa_analyzer.data import load_own_chat

MOVE_DATE = settings.MOVE_DATE
EVENT_DATE = settings.TEST_DATE  # positive pregnancy test
BABY_DATE = settings.BIRTH_DATE
PREGNANCY_START = settings.PREGNANCY_START

# Compiled, as the notebooks use them (`.str.contains(pattern)` works either way).
SYMPTOM_PATTERNS = {
    name: re.compile(p) for name, p in settings.SYMPTOM_PATTERNS.items()
}

SIDE_COLORS = {
    "before": "#7f7f7f",
    "after": PlotSettings().highlight_color,
}  # grey-first, 02.2

FIGURES = (
    next(
        p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
    )
    / "figures"
)
FIGURES.mkdir(exist_ok=True)


def save(fig, name: str) -> None:
    """PNG for the report, SVG for anything that needs to scale."""
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight")


def canonical_author(name: str) -> str:
    """Louis appears under three names in the export (a contact rename mid-chat);
    group them into one author before any per-person distribution."""
    if "louis" in str(name).lower():
        return "Louis"
    return name


def load_own() -> pd.DataFrame:
    """The chat export, one author name per person, plus message length in characters."""
    own = load_own_chat()
    own["timestamp"] = pd.to_datetime(own["timestamp"])
    own["author"] = own["author"].apply(canonical_author)
    own["n_chars"] = own["message"].str.len()
    return own


def local_chat(own: pd.DataFrame) -> pd.DataFrame:
    """Timestamps arrive labelled UTC but are the phone's local clock (almost nothing between
    02:00 and 05:00), so dropping the label gives local wall-clock time -- same as 4.6.4."""
    chat = own[["timestamp", "author", "message"]].copy()
    chat["ts_local"] = chat["timestamp"].dt.tz_convert(None)
    return chat.sort_values("ts_local").reset_index(drop=True)

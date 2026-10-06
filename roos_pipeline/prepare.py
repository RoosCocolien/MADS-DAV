"""Stage 1 — prepare: the raw export, cleaned into one row per message.

    export (csv) -> CanonicalAuthor -> LocalTimestamp -> SortValues -> 1_messages.parquet

Nothing here knows about languages, topics or pregnancies; it only makes the frame
trustworthy: one name per person, one timestamp column that every later stage can
compare against a plain date.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd
from goad_toolkit.datatransforms import Pipeline, SortValues, TransformBase
from loguru import logger

from roos_pipeline.settings import Paths
from wa_analyzer.data import load_own_chat

EXPECTED_AUTHORS = 2  # Roos and Louis


class CanonicalAuthor(TransformBase):
    """Merge several display names of one person into a single author.

    WhatsApp keys authors by display name, so one person who renamed themselves (or
    wrote from a second phone) shows up as several authors. The mapping is explicit
    rather than a substring match: `"louis" in name` would also swallow a future
    "Louise". Must run before anything that groups or hashes by author.
    """

    def transform(
        self, data: pd.DataFrame, aliases: dict[str, str], column: str = "author"
    ) -> pd.DataFrame:
        renamed = data[column].isin(aliases.keys()).sum()
        data[column] = data[column].replace(aliases)
        logger.info(
            f"{self.name}: merged {renamed:,} messages; "
            f"authors now {sorted(data[column].unique())}"
        )
        return data


class LocalTimestamp(TransformBase):
    """Parse the exported timestamp string once, into the phone's wall-clock time.

    The export labels timestamps UTC, but they are the phone's local clock (almost
    nothing between 02:00 and 05:00, see roos_04.6 §4.6.4). Dropping the label keeps
    that wall-clock time, and a timezone-naive column compares directly against the
    plain dates in `settings`.
    """

    def transform(
        self, data: pd.DataFrame, column: str = "timestamp", feature: str = "ts_local"
    ) -> pd.DataFrame:
        data[feature] = pd.to_datetime(data[column], utc=True).dt.tz_convert(None)
        return data


def load_author_aliases(config: Path) -> dict[str, str]:
    """Read `[author_aliases]` from `config.toml`: display name -> canonical name.

    Lives in the (gitignored) config rather than in code, because the names are
    personal data and specific to one export.
    """
    if not config.exists():
        logger.warning(f"No {config}; authors are left as-is.")
        return {}
    with config.open("rb") as f:
        aliases = tomllib.load(f).get("author_aliases", {})
    if not aliases:
        logger.warning(f"No [author_aliases] in {config}; authors are left as-is.")
    return aliases


def build_prepare_pipeline(aliases: dict[str, str]) -> Pipeline:
    """CanonicalAuthor first, so no later step ever sees the split-up names."""
    return (
        Pipeline()
        .add(CanonicalAuthor, aliases=aliases)
        .add(LocalTimestamp)
        .add(SortValues, column="ts_local")
    )


def check_messages(data: pd.DataFrame) -> None:
    """Fail loudly on the mistakes that would otherwise only show up as a wrong chart.

    A new export can bring a new display name (a third phone, a rename). Without this
    check every per-person number would silently split one person in two.
    """
    authors = sorted(data["author"].unique())
    if len(authors) != EXPECTED_AUTHORS:
        raise ValueError(
            f"Expected {EXPECTED_AUTHORS} authors after merging aliases, got "
            f"{authors}. Add the new display name to [author_aliases] in config.toml."
        )
    if data["ts_local"].isna().any():
        raise ValueError("Some timestamps could not be parsed.")


def run(paths: Paths, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load the export (or `raw`, in tests), clean it, and write 1_messages.parquet."""
    raw = load_own_chat() if raw is None else raw
    messages = build_prepare_pipeline(load_author_aliases(paths.config)).apply(raw)
    check_messages(messages)
    paths.processed.mkdir(parents=True, exist_ok=True)
    messages.to_parquet(paths.messages)
    logger.success(f"prepare: {len(messages):,} messages -> {paths.messages}")
    return messages

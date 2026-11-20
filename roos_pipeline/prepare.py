"""Stage 1 — prepare: the raw export, cleaned into one row per message.

    export (csv) -> CanonicalAuthor -> LocalTimestamp -> SortValues -> 1_messages.parquet
                 -> UnmergeMessages -> CanonicalAuthor -> Filter   -> 1_unmerged.parquet

The second file is for the les-6 and les-7 figures: the export sometimes glues
several messages into one row, and those notebooks split them back apart. Les 2-5
keep `1_messages`, so their numbers stay the ones they were made with.

Nothing here knows about languages, topics or pregnancies; it only makes the frame
trustworthy: one name per person, one timestamp column that every later stage can
compare against a plain date.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pandas as pd
from goad_toolkit.datatransforms import Filter, Pipeline, SortValues, TransformBase
from loguru import logger

from roos_pipeline.settings import CATEGORIES, PATTERNS, Categories, Paths
from wa_analyzer.data import load_own_chat


class UnmergeMessages(TransformBase):
    """Split a row that holds several messages into one row per message.

    After an "afbeelding weggelaten" line the export sometimes runs the next messages into
    the same row, each behind its own "[dd-mm-yyyy, hh:mm:ss] Name: " header; 409 of those
    headers carry a different day than the row (counted 2026-10-30). Lesson 6 cut them off
    (`StripQuotedReply`); here they are kept, because a place in them was mentioned on the
    day in *their* header. If a header is a quote of an older message instead, its day was
    already counted then, and a day is only counted once per place, so nothing is doubled.
    """

    def transform(
        self,
        data: pd.DataFrame,
        column: str = "message",
        time_format: str = "%d-%m-%Y, %H:%M:%S",
        header: str = PATTERNS.embedded_header,
    ) -> pd.DataFrame:
        rows = []
        for when, author, text in zip(
            data["ts_local"], data["author"], data[column].fillna(""), strict=True
        ):
            parts = re.split(header, text)
            rows.append((when, author, parts[0]))
            for stamp, name, piece in zip(parts[1::3], parts[2::3], parts[3::3]):
                rows.append((pd.to_datetime(stamp, format=time_format), name, piece))
        out = pd.DataFrame(rows, columns=["ts_local", "author", column])
        out[column] = out[column].str.replace("‎", "", regex=False).str.strip()
        return out.sort_values("ts_local", kind="stable").reset_index(drop=True)


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
        data = data.copy()
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
        data = data.copy()
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


def build_unmerge_pipeline(aliases: dict[str, str]) -> Pipeline:
    """Split glued rows apart; their headers bring back old names, so merge them again."""
    return (
        Pipeline()
        .add(UnmergeMessages)
        .add(CanonicalAuthor, aliases=aliases)
        .add(Filter, expr="message != ''")
    )


def check_messages(data: pd.DataFrame, categories: Categories = CATEGORIES) -> None:
    """Fail loudly on the mistakes that would otherwise only show up as a wrong chart.

    A new export can bring a new display name (a third phone, a rename). Without this
    check every per-person number would silently split one person in two.
    """
    authors = sorted(data["author"].unique())
    if len(authors) != len(categories.authors):
        raise ValueError(
            f"Expected {len(categories.authors)} authors after merging aliases, got "
            f"{authors}. Add the new display name to [author_aliases] in config.toml."
        )
    if data["ts_local"].isna().any():
        raise ValueError("Some timestamps could not be parsed.")


def run(paths: Paths, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load the export (or `raw`, in tests), clean it, write both stage-1 files."""
    raw = load_own_chat() if raw is None else raw
    aliases = load_author_aliases(paths.config)
    messages = build_prepare_pipeline(aliases).apply(raw)
    check_messages(messages)
    unmerged = build_unmerge_pipeline(aliases).apply(messages)
    check_messages(unmerged)
    paths.processed.mkdir(parents=True, exist_ok=True)
    messages.to_parquet(paths.messages)
    unmerged.to_parquet(paths.unmerged)
    logger.success(
        f"prepare: {len(messages):,} messages -> {paths.messages}; "
        f"{len(unmerged):,} after splitting -> {paths.unmerged}"
    )
    return messages

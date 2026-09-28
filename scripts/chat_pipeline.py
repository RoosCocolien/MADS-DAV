"""Roos's own chat, from raw export to the two figures the story rests on.

`roos_01.4-language-exploration.ipynb` derived every step here cell by cell; this
module is where the steps the final figures depend on live afterwards, so the figures
can be rebuilt in one command instead of by re-running a notebook top to bottom:

    uv run python scripts/chat_pipeline.py

Load -> transform (one `Pipeline`) -> aggregate -> plot -> save. Each stage is also
importable on its own, e.g. to reuse the cleaned frame in a notebook:

    from scripts.chat_pipeline import build_chat_pipeline, load_author_aliases
    from wa_analyzer.data import load_own_chat

    chat = build_chat_pipeline(load_author_aliases()).apply(load_own_chat())
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py3langid as langid
from goad_toolkit.datatransforms import Pipeline, RegexFeature, TransformBase
from loguru import logger
from matplotlib.figure import Figure

from wa_analyzer.data import CONFIG, ROOT, load_own_chat

IMG_DIR = ROOT / "img"

# The story's timeline — Roos's own dates, not derived from the data.
PREGNANCY_DATE = pd.Timestamp("2024-06-01")
MOVE_DATE = pd.Timestamp("2024-11-01")
BIRTH_DATE = pd.Timestamp("2025-05-01")
MILESTONES = [
    (PREGNANCY_DATE, "pregnancy", "--"),
    (MOVE_DATE, "move", ":"),
    (BIRTH_DATE, "baby born", "-."),
]

LANGUAGES = ["en", "nl", "fr"]

# Only the two categories the final figures use; the full set of eight stays in 01.4.
CATEGORY_PATTERNS: dict[str, str] = {
    "child": (
        r"(?i)("
        r"benji|b[ée]b[ée]|\bbaby\b|cr[eè]che|kinderdagverblijf|opvang|daycare|partou|basker|kinderdagopvang|"
        r"r[eé]cuperer|r[eé]cupes|prendre soin de|"
        r"\benfant\b|biberon|bottle[- ]?feeding|sucette|fopspeen|speentje|"
        r"pacifier|dummy|allaiter|donner les? seins?|borstvoeding|breastfeeding|t[ée]t[ée]e|"
        r"luiers?|diapers?|couches?|doekje|hydrophile|vitamines|hydrofiele doek|\bwipe\b|"
        r"kinderwagen|poussette|stroller|pram|la remorque|bike trailer|"
        r"consultatiebureau|kraamzorg|zwangerschap|grossesse|pregnan(cy|t)|"
        r"verloskundige|sage-femme|midwife|kolven|[ée]chographie"
        r")"
    ),
    "climbing": (
        r"(?i)("
        r"escalad(er|e)|salle de grimpe|grimpe|grimper|\bclimb(ing)?\b|climbing gym|"
        r"klimmen|klimmuur( centraal)?|\bboulder\b|\bbloc\b|\bvoie\b|"
        r"arkose|mountain network|\bncc\b|neoliet|\b[4-8][abc]\+?\b"
        r")"
    ),
}


# --- transforms: one row per message in, one row per message out -----------------


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


class NaiveTimestamp(TransformBase):
    """Parse the exported timestamp string once into a timezone-naive datetime.

    The export stores timestamps as UTC-offset strings; every later step compares
    against plain `pd.Timestamp` dates, which only works once the timezone is gone.
    """

    def transform(
        self,
        data: pd.DataFrame,
        column: str = "timestamp",
        feature: str = "timestamp_naive",
    ) -> pd.DataFrame:
        data[feature] = pd.to_datetime(data[column], utc=True).dt.tz_convert(None)
        return data


class LanguageFeature(TransformBase):
    """Detect the language of a text column with `py3langid`.

    Restricting the candidate set (`languages`) is what makes this usable on short,
    sometimes-incomplete chat messages: without it, `py3langid` picks among ~97
    languages and misfires far more often on a five-word message.
    """

    def transform(
        self,
        data: pd.DataFrame,
        column: str,
        languages: list[str],
        feature: str = "language",
        min_chars: int = 3,
    ) -> pd.DataFrame:
        langid.set_languages(languages)

        def detect(msg: str) -> str:
            if len(msg.strip()) < min_chars:
                return "unknown"
            return langid.classify(msg)[0]

        data[feature] = data[column].fillna("").map(detect)
        return data


class SplitAtDate(TransformBase):
    """Label each message as falling before or after one date."""

    def transform(
        self,
        data: pd.DataFrame,
        date: pd.Timestamp,
        labels: tuple[str, str] = ("before", "after"),
        column: str = "timestamp_naive",
        feature: str = "period",
    ) -> pd.DataFrame:
        data[feature] = np.where(data[column] < date, labels[0], labels[1])
        return data


def build_chat_pipeline(aliases: dict[str, str]) -> Pipeline:
    """Assemble every per-message step the two final figures need, in order.

    Order matters in one place: `CanonicalAuthor` first, so no later step ever sees
    the split-up author names.
    """
    pipeline = Pipeline()
    pipeline.add(CanonicalAuthor, aliases=aliases)
    pipeline.add(NaiveTimestamp)
    pipeline.add(LanguageFeature, column="message", languages=LANGUAGES)
    for category, pattern in CATEGORY_PATTERNS.items():
        pipeline.add(
            RegexFeature,
            name=f"{category}_words",
            column="message",
            pattern=pattern,
            feature=f"n_{category}_words",
            mode="count",
        )
    pipeline.add(
        SplitAtDate,
        name="move_period",
        date=MOVE_DATE,
        labels=("before move", "after move"),
        feature="move_period",
    )
    return pipeline


def load_author_aliases(config: Path = CONFIG) -> dict[str, str]:
    """Read `[author_aliases]` from `config.toml`: display name -> canonical name.

    Lives in the (gitignored) config rather than in this file, because the names are
    personal data and specific to one export.
    """
    with config.open("rb") as f:
        aliases = tomllib.load(f).get("author_aliases", {})
    if not aliases:
        logger.warning(f"No [author_aliases] in {config}; authors are left as-is.")
    return aliases


# --- aggregations: change the grain from messages to weeks / periods -------------


def weekly_category_share(data: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Per week, the share of all messages that mention each category at least once.

    A share rather than a count: message volume itself drops after the move, and a
    raw count would re-tell that drop instead of a change in topic.
    """
    week = data["timestamp_naive"].dt.to_period("W").dt.to_timestamp()
    has_category = pd.DataFrame(
        {c: data[f"n_{c}_words"].gt(0) for c in categories}
    ).assign(week=week)
    n_messages = week.value_counts().rename_axis("week").rename("n_messages")
    return (
        has_category.groupby("week")[categories]
        .sum()
        .reset_index()
        .melt(id_vars="week", var_name="category", value_name="n_with_category")
        .merge(n_messages, on="week")
        .assign(share=lambda d: d["n_with_category"] / d["n_messages"])
    )


def language_share_by_period(
    data: pd.DataFrame, period_col: str = "move_period"
) -> pd.DataFrame:
    """Percentage of each language among *classified* messages, per period.

    `"unknown"` (too short to classify) is left out of the denominator, so the bars
    within one period add up to 100%.
    """
    known = data[data["language"] != "unknown"]
    counts = known.groupby([period_col, "language"]).size().rename("n").reset_index()
    counts["pct"] = counts.groupby(period_col)["n"].transform(
        lambda s: 100 * s / s.sum()
    )
    return counts


# --- plots -----------------------------------------------------------------------


def plot_child_vs_climbing(weekly: pd.DataFrame, smooth_weeks: int = 4) -> Figure:
    """Child vs. climbing share of weekly messages, raw in grey, trend in colour."""
    colors = {"child": "#0072B2", "climbing": "#D55E00"}
    focus = weekly[weekly["category"].isin(colors)].copy()
    focus["share_trend"] = focus.groupby("category")["share"].transform(
        lambda s: s.rolling(smooth_weeks, center=True, min_periods=1).mean()
    )
    n_by_category = focus.groupby("category")["n_with_category"].sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvspan(
        mdates.date2num(PREGNANCY_DATE),
        mdates.date2num(BIRTH_DATE),
        color="#F0E442",
        alpha=0.15,
        zorder=0,
    )
    for category, color in colors.items():
        cat_data = focus[focus["category"] == category]
        ax.plot(cat_data["week"], cat_data["share"], color="lightgray", lw=1, zorder=1)
        ax.plot(
            cat_data["week"],
            cat_data["share_trend"],
            color=color,
            lw=2.5,
            label=f"{category} ({smooth_weeks}-week trend)",
            zorder=2,
        )
    for date, _, style in MILESTONES:
        ax.axvline(
            mdates.date2num(date), color="grey", linestyle=style, linewidth=1, alpha=0.7
        )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("share of weekly messages")
    ax.set_title("Child vs. climbing: a trade in the conversation")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.spines[["top", "right"]].set_visible(False)
    n_text = " · ".join(f"{cat}: n={n:,} messages" for cat, n in n_by_category.items())
    fig.text(0.5, -0.03, f"Based on {n_text}", ha="center", fontsize=9, color="dimgray")
    fig.tight_layout()
    return fig


def plot_praat_nederlands(lang_pct: pd.DataFrame) -> Figure:
    """Language mix before vs. after the move, Dutch highlighted."""
    lang_colors = {"en": "#C8C8C8", "nl": "#D55E00", "fr": "#C8C8C8"}
    period_labels = {
        "before move": "Voor de verhuizing",
        "after move": "Na de verhuizing",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    for ax, period in zip(axes, period_labels):
        subset = (
            lang_pct[lang_pct["move_period"] == period]
            .set_index("language")
            .reindex(LANGUAGES)
            .fillna(0)
            .reset_index()
        )
        bars = ax.bar(
            subset["language"],
            subset["pct"],
            color=[lang_colors[lang] for lang in subset["language"]],
        )
        ax.bar_label(
            bars, labels=[f"{v:.0f}%" for v in subset["pct"]], padding=3, fontsize=10
        )
        ax.set_title(period_labels[period])
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% van geclassificeerde berichten")
    fig.suptitle("Praat Nederlands met mij", fontsize=18, fontweight="bold", y=1.04)
    fig.text(
        0.5,
        0.95,
        "Na de verhuizing naar Nederland, groeit Nederlands, "
        "maar Frans blijft de dominante taal.",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def save_fig(fig: Figure, name: str) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    path = IMG_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    logger.success(f"Saved {path}")


def main() -> None:
    chat = build_chat_pipeline(load_author_aliases()).apply(load_own_chat())
    weekly = weekly_category_share(chat, list(CATEGORY_PATTERNS))
    save_fig(plot_child_vs_climbing(weekly), "final_child_vs_climbing.png")
    save_fig(
        plot_praat_nederlands(language_share_by_period(chat)),
        "final_praat_nederlands_met_mij.png",
    )
    logger.success("All done.")


if __name__ == "__main__":
    main()

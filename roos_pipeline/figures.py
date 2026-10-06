"""Stage 4 — figures: read the stage-3 tables, draw, save. No computing on messages.

    3_language_share.parquet  -> praat_nederlands_panels   (roos_01.4, code cell 13)
                              -> praat_nederlands_bars     (roos_02.5, code cell 5)
    3_weekly_topics.parquet   -> baby_is_taking_over x2    (roos_03.3, code cells 6, 7)
    3_symptom_stats.json      -> prepare_and_plan          (roos_04.6d, code cell 11)

Each figure is one function: settings (titles, labels, colours) in, `Figure` out. The
only calculations left here are presentation choices: the rolling average (how
smooth a line looks) and evaluating a fitted curve at whole numbers.
"""

from __future__ import annotations

import json

import pandas as pd
from goad_toolkit.datatransforms import Pipeline, RollingAvg
from goad_toolkit.visualizer import PlotSettings
from loguru import logger
from matplotlib.figure import Figure

from roos_pipeline import settings
from roos_pipeline.aggregate import nbinom_from
from roos_pipeline.plots import (
    LabelledBarPlot,
    PeakLabelledCurves,
    PeriodPanelsPlot,
    TopicTrendPlot,
)
from roos_pipeline.settings import Paths

FOOTNOTE = {"fontsize": 8.5, "color": "dimgray"}


def save(fig: Figure, name: str, paths: Paths) -> None:
    """PNG for the report, SVG for anything that needs to scale."""
    paths.figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(paths.figures / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    logger.info(f"saved {paths.figures / name}.png/.svg")


# --- comparing categories ----------------------------------------------------------


def praat_nederlands_panels(lang: pd.DataFrame) -> Figure:
    """Language mix before vs. after the move, one panel each, Dutch highlighted."""
    panel_settings = PlotSettings(
        figsize=(9, 5),
        title="",
        xlabel="",
        # Shared y-axis, so only the left panel needs the label.
        subplot_ylabels=["% van geclassificeerde berichten", ""],
        subplot_titles=[settings.PERIOD_LABELS_NL[p] for p in settings.PERIOD_ORDER],
        sharey=True,
        highlight=["nl"],
        highlight_color=settings.SPORT_ORANGE,
        base_color=settings.LIGHT_GREY,
    )
    fig, _ = PeriodPanelsPlot(panel_settings).plot(
        n_plots=2,
        data=lang,
        panel="move_period",
        x="language",
        y="pct",
        panels=settings.PERIOD_ORDER,
        order=settings.LANGUAGES,
    )
    fig.suptitle(
        "Praat Nederlands met mij\n",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.89,
        "Na de verhuizing naar Nederland, groeit Nederlands, "
        "maar Frans blijft de dominante taal.",
        ha="center",
        fontsize=11,
    )
    return fig


def praat_nederlands_bars(lang: pd.DataFrame) -> Figure:
    """The same numbers on one axes, before/after as hue: the 02.5 rework after the
    teacher's comment that two panels mean comparing 2x3 things."""
    bar_settings = PlotSettings(
        figsize=(8, 5),
        title="Praat Nederlands met mij",
        xlabel="language",
        ylabel="% of classified messages",
    )
    fig, ax = LabelledBarPlot(bar_settings).plot(
        data=lang,
        x="language",
        y="pct",
        hue="move_period",
        order=settings.LANGUAGES,
        hue_order=settings.PERIOD_ORDER,
        palette={"before move": "#cccccc", "after move": settings.SPORT_ORANGE},
    )
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="")
    n = lang.groupby("move_period")["n"].sum()
    fig.text(
        0.5,
        -0.05,
        f"Based on {n['before move']:,} classified messages before the move, "
        f"{n['after move']:,} after (messages under {settings.MIN_CHARS} characters "
        "excluded -- too short to reflect a language choice either way).",
        ha="center",
        **FOOTNOTE,
    )
    return fig


# --- time series -------------------------------------------------------------------


def p_text(n_reaching: int, p: float) -> str:
    """'p ≤ 0.0001' when no shuffle reached the real value (p is then only a bound)."""
    return f"p ≤ {p:.4f}" if n_reaching == 0 else f"p ≈ {p:.4f}"


def balance_text(stats: dict) -> str:
    """The shuffle test and the time-order check, as two sentences under the chart."""
    sport = stats["sport"]
    before, after = 100 * stats["mean_gap_before"], 100 * stats["mean_gap_after"]
    first = (
        f"Child minus {sport}, in percentage points of weekly messages: {before:+.1f} "
        f"before the positive test, {after:+.1f} after ({stats['n_before']} and "
        f"{stats['n_after']} weeks; the week of the test left out).\nShuffling the "
        f"before/after label over all weeks ({stats['n_shuffle']:,} times) produced "
        f"a shift this large {stats['n_shuffles_reaching_shift']} times "
        f"({p_text(stats['n_shuffles_reaching_shift'], stats['p_value_greater'])})."
    )
    if stats["n_runs"] == 0:
        second = (
            f"Too few weeks before the test for a run of {stats['run_weeks']} "
            "consecutive weeks: no time-order check."
        )
    else:
        second = (
            f"Keeping the order of time: {stats['n_runs_reaching_after']} of the "
            f"{stats['n_runs']} runs of {stats['run_weeks']} consecutive weeks before "
            f"the test reached the after-test average (highest run "
            f"{100 * stats['max_run_gap']:+.1f})."
        )
    return f"{first}\n{second}"


def baby_is_taking_over(
    weekly: pd.DataFrame, colors: dict[str, str], balance: dict
) -> Figure:
    """Weekly share of messages per topic, with the test and the birth marked, and the
    statistical test of child vs. sport (computed in aggregate.py) underneath."""
    trend = Pipeline()
    for topic in colors:
        trend.add(RollingAvg, column=topic, window=settings.SMOOTH_WEEKS, rename=True)
    weekly_trend = trend.apply(weekly)

    trend_settings = PlotSettings(
        figsize=(12, 5),
        title="Baby is taking over",
        xlabel="",
        ylabel="share of weekly messages",
        xtick_rotation=45,
    )
    fig, _ = TopicTrendPlot(trend_settings).plot(
        weekly=weekly,
        trend=weekly_trend,
        x="week",
        colors=colors,
        events={
            "positive pregnancy test": str(settings.TEST_DATE.date()),
            "baby is born": str(settings.BIRTH_DATE.date()),
        },
        smooth_weeks=settings.SMOOTH_WEEKS,
    )
    # Every weekly share is a share of that week's messages: say how many that is.
    n = weekly["n_messages"]
    mentions = " · ".join(f"{t}: {int(weekly[f'{t}_n'].sum()):,}" for t in colors)
    fig.text(
        0.5,
        -0.08,
        f"Based on {int(n.sum()):,} messages over {len(weekly)} weeks "
        f"(median {int(n.median())} per week, fewest {int(n.min())}). "
        f"Messages mentioning each topic: {mentions}.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.text(0.5, -0.12, balance_text(balance), ha="center", va="top", **FOOTNOTE)
    return fig


# --- distributions -----------------------------------------------------------------


def prepare_and_plan(stats: dict) -> Figure:
    """Normal weeks vs. hard weeks as two fitted negative binomials, shuffle test below."""
    ratio = stats["mean_hard"] / stats["mean_normal"]
    # The subtitle says "three times" in words: if the data upstream ever changes,
    # this stops the pipeline instead of printing a claim the numbers no longer make.
    if round(ratio) != 3:
        raise ValueError(f"Subtitle says 'three times', data now say {ratio:.1f}x")

    curves = pd.DataFrame({"k": range(stats["max_count"] + 3)})
    curves["normal"] = nbinom_from(stats["normal_mu"], stats["normal_r"]).pmf(
        curves["k"]
    )
    curves["hard"] = nbinom_from(stats["hard_mu"], stats["hard_r"]).pmf(curves["k"])

    both_settings = PlotSettings(
        figsize=(10, 6),
        title="",
        xlabel="messages mentioning nausea or fatigue, per week",
        ylabel="probability: share of weeks with exactly this many",
    )
    fig, ax = PeakLabelledCurves(both_settings).plot(
        data=curves,
        x="k",
        curves={
            "normal": (settings.GREY, "normal weeks"),
            "hard": (settings.CRIMSON, "hard weeks"),
        },
    )
    ax.set_xlim(-0.5, curves["k"].max() + 0.5)
    ax.set_title(
        "In the hard weeks, nausea and fatigue came up three times as often: "
        f"{stats['mean_hard']:.0f} messages a week instead of {stats['mean_normal']:.0f}",
        loc="left",
        fontsize=12,
        color="dimgray",
    )
    fig.suptitle(
        "How to survive a pregnancy: prepare and plan.",
        x=0.125,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.02,
        f"Grey: {stats['n_normal']} normal long-distance weeks before the pregnancy "
        f"(average {stats['mean_normal']:.1f} a week). Crimson: the {stats['n_hard']} "
        f"hard weeks, pregnancy weeks 5-14 (average {stats['mean_hard']:.1f}).\n"
        "Both lines: fitted negative binomial.",
        va="top",
        **FOOTNOTE,
    )
    fig.text(
        0.01,
        -0.08,
        "Chance alone does not explain this gap: when the 'hard' label was shuffled at "
        f"random over all {stats['n_weeks_shuffled']} weeks ({stats['n_shuffle']:,} "
        f"times),\na gap as large as the real {stats['gap_observed']:+.1f} symptom "
        f"messages a week occurred {stats['n_shuffles_reaching_gap']} times "
        f"(p ≤ {stats['p_value_greater']:.4f}).",
        va="top",
        **FOOTNOTE,
    )
    return fig


def run(paths: Paths) -> None:
    """Read the stage-3 files and write all five figures."""
    lang = pd.read_parquet(paths.language_share)
    weekly = pd.read_parquet(paths.weekly_topics)
    stats = json.loads(paths.symptom_stats.read_text())
    balance = json.loads(paths.topic_stats.read_text())

    save(praat_nederlands_panels(lang), "1_praat_nederlands_panels", paths)
    save(praat_nederlands_bars(lang), "2_praat_nederlands_bars", paths)
    for i, sport in [(3, "climbing"), (4, "sports")]:
        colors = {"child": settings.CHILD_BLUE, sport: settings.SPORT_ORANGE}
        fig = baby_is_taking_over(weekly, colors, balance[sport])
        save(fig, f"{i}_baby_taking_over_{sport}", paths)
    save(prepare_and_plan(stats), "5_prepare_and_plan", paths)
    logger.success(f"figures: 5 figures -> {paths.figures}")

"""Stage 4 — figures: read the stage-3 tables, draw, save. No computing on messages.

    3_language_share.parquet  -> praat_nederlands_panels   (roos_01.4, code cell 13)
                              -> praat_nederlands_bars     (roos_02.5, code cell 5)
    3_weekly_topics.parquet   -> baby_is_taking_over x2    (roos_03.3, code cells 6, 7)
    3_symptom_stats.json      -> prepare_and_plan          (roos_04.6d, code cell 11)
    3_weekly_french.parquet   -> familiar_language         (roos_05.6, code cell 14)
    3_french_stats.json
    3_home_shares.parquet     -> onderweg_naar_thuis       (roos_07.4, code cell 24)
    3_home_stats.json
    3_style_docs.parquet      -> frans_van_roos_en_louis   (roos_06.6, code cell 19)
    3_style_means.parquet
    3_style_stats.json

Each figure is one function: settings (titles, labels, colours) in, `Figure` out. The
only calculations left here are presentation choices: the rolling average (how
smooth a line looks) and evaluating a fitted curve at whole numbers.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from goad_toolkit.datatransforms import Pipeline, RollingAvg
from goad_toolkit.visualizer import PlotSettings, RegPlot
from loguru import logger
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator, PercentFormatter

from roos_pipeline import settings
from roos_pipeline.aggregate import nbinom_from
from roos_pipeline.plots import (
    HomeAwayPlot,
    LabelledBarPlot,
    PeakLabelledCurves,
    PeriodPanelsPlot,
    StyleMapPlot,
    TopicTrendPlot,
)
from roos_pipeline.settings import Paths

# `Any`, not the inferred `float | str`: ty checks a `**` dict's value type against
# every keyword of `fig.text`, including `fontdict`, which only takes a dict.
FOOTNOTE: dict[str, Any] = {"fontsize": 8.5, "color": "dimgray"}


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


# --- relationships -----------------------------------------------------------------

LANGUAGE_NAMES = {"en": "English", "nl": "Dutch", "fr": "French"}


def language_caption(stats: dict) -> str:
    """How many of Louis's messages the dots rest on, and their language mix.

    Percentages are over the classified messages, so they add up to 100%; the
    "unknown" ones are counted separately.
    """
    counts = stats["language_counts"]
    known = sum(counts[code] for code in LANGUAGE_NAMES)
    shares = ", ".join(
        f"{100 * counts[code] / known:.0f}% {name}"
        for code, name in LANGUAGE_NAMES.items()
    )
    return (
        f"Based on {stats['n_messages']:,} messages from Louis over "
        f"{stats['n_weeks']} weeks: {shares}\n"
        f"(of the {known:,} messages long enough to classify; "
        f"{counts['unknown']:,} messages under {settings.MIN_CHARS} characters "
        f"have no language but do count for words per message)."
    )


def familiar_language(weekly: pd.DataFrame, stats: dict) -> Figure:
    """One dot per long-distance week: Louis's French share against his median message
    length, with the log-of-x line and the within-week test underneath."""
    # The footnote says "p < 0.001" and "fewer" in words: if the data upstream ever
    # change, this stops the pipeline instead of printing a claim the test no longer makes.
    if not (stats["wilcoxon_p_less"] < 0.001 and stats["median_difference"] < 0):
        raise ValueError(
            "The test no longer supports the line under the graph: rewrite it."
        )

    final = PlotSettings(
        figsize=(9, 6),
        title="",
        xlabel="share of messages in French that week",
        ylabel="words per message (median that week)",
    )
    # logx: median words = a + b * ln(share), the line chosen in roos_05.6 §5.6.5.
    fig, ax = RegPlot(final).plot(
        data=weekly,
        x="french_share",
        y="median_words",
        logx=True,
        ci=None,
        color=settings.DARK_GREY,
        scatter_kws={"color": settings.LONG_DISTANCE_ORANGE, "alpha": 0.75},
    )
    ax.set(xlabel=final.xlabel, ylabel=final.ylabel)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.text(
        0.98,
        0.80,
        "line: every doubling of French\n"
        f"= {abs(stats['words_per_doubling']):.1f} fewer words per message",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=settings.DARK_GREY,
    )

    fig.subplots_adjust(top=0.84)
    fig.suptitle(
        "Familiar language needs fewer words", fontsize=18, fontweight="bold", y=0.98
    )
    fig.text(
        0.5,
        0.905,
        "While we lived apart, Louis wrote shorter messages when he wrote in French.",
        ha="center",
        fontsize=11,
    )

    first_day = pd.Timestamp(stats["first_day"])
    last_week = pd.Timestamp(stats["last_week"])
    footnote = "\n".join(
        [
            f"Each dot is one week we lived apart ({first_day:%b %Y} – "
            f"{last_week:%b %Y}): what share of Louis's messages that week was "
            "French, and how many words his middle message had.",
            language_caption(stats),
            "Within the same week, his French messages were shorter than his "
            f"English and Dutch ones in {stats['n_weeks_below']} of "
            f"{stats['n_weeks_compared']} weeks (typically "
            f"{abs(stats['median_difference']):.0f} words fewer; Wilcoxon "
            "signed-rank, p < 0.001).",
        ]
    )
    fig.text(0.5, -0.02, footnote, ha="center", va="top", linespacing=1.5, **FOOTNOTE)
    return fig


# --- Van onderweg naar thuis (lesson 7) ---------------------------------------------


def dutch_thousands(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def home_caption(stats: dict) -> str:
    """The three paragraphs under figure 7: dots and lines, themes, the shuffle test."""
    by_message, by_day = stats["messages"], stats["days"]
    n = dutch_thousands(stats["n_shuffle"])
    p_floor = f"{1 / (stats['n_shuffle'] + 1):.4f}".replace(".", ",")
    against = stats["against"]
    return (
        "Elke donkere stip is een stad. Hoe verder naar rechts, hoe hoger de "
        "thuis-score: het deel van de thema's in de gesprekken over die stad dat over "
        "thuis met kind gaat, tegenover spontaan op pad. Een lijn verbindt een stad met "
        "een thema: hoe dikker, hoe vaker dat thema voorkwam in een gesprekje over de "
        f"stad (het bericht dat de stad noemt plus de {settings.HOME_WINDOW} berichten "
        f"erna). Lijnen onder {settings.HOME_MIN_SHARE:.0%} zijn weggelaten.\n\n"
        "De zes thema's vallen in twee groepen: spontaan op pad (reizen, klimmen & "
        "sport, eten & uitgaan) en thuis met kind (Benji & kind, wonen, familie). Werk "
        "en vakantie tellen niet mee. Een thema is gevonden met een woordenlijst in het "
        "Nederlands, Frans en Engels, bijvoorbeeld trein, TGV en gare; klimmen, "
        "escalade en hardlopen; restaurant en apéro; Benji, crèche en luiers; huis, "
        "appartement en verhuizen; ouders, oma en mes parents. Een woord kan ook iets "
        "anders betekenen.\n\n"
        f"Schudtest. De thuis-score van {stats['place']} ligt "
        f"{by_message['gap_observed'] * 100:.0f} procentpunt hoger dan die van "
        f"{', '.join(against[:-1])} en {against[-1]} samen. We hebben {n} keer "
        "willekeurig gehusseld welke stad bij welk gesprekje hoort. In "
        f"{by_message['n_shuffles_reaching_gap']} van de {n} keer kwam er een even "
        f"groot verschil uit (p < {p_floor}); met een dag als eenheid in plaats van een "
        f"bericht {by_day['n_shuffles_reaching_gap']} keer. Het verschil is dus "
        "vrijwel zeker geen toeval; waarom het er is, zegt de test niet."
    )


def onderweg_naar_thuis(shares: pd.DataFrame, stats: dict) -> Figure:
    """Figure 7: the places between the on-the-road and the home themes."""
    # "vrijwel zeker geen toeval" in words: stop if the test stops saying so.
    for unit in ("messages", "days"):
        if not (
            stats[unit]["gap_observed"] > 0 and stats[unit]["p_value_greater"] < 0.01
        ):
            raise ValueError(
                f"The shuffle test per {unit} no longer supports the caption: rewrite it."
            )
    home_plot = PlotSettings(
        figsize=(16, 11),
        title="De reis van onze relatie: van spontaan op pad naar samen thuis.",
    )
    fig, _ = HomeAwayPlot(home_plot).plot(
        table=shares,
        subtitle=(
            "Tussen Parijs, Amsterdam en Utrecht ging het over reizen, sport en "
            "uitgaan. Sinds de komst van Benji\ngaat het in Nieuwegein over wonen en ons "
            "kind, met hulp van opa en oma in Zoetermeer."
        ),
        caption=home_caption(stats),
    )
    return fig


# --- Roos en Louis groeien naar elkaars Frans toe (lesson 6) -------------------------


def p_short(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def style_texts(stats: dict) -> dict[str, str]:
    """Every sentence on figure 8. The axis examples come from the loadings (06.6)."""
    similarity = stats["similarity"]
    first_q, last_q = next(iter(similarity)), list(similarity)[-1]
    roos, louis = stats["roos_trend"], stats["louis_trend"]
    first, last = pd.Timestamp(stats["first_day"]), pd.Timestamp(stats["last_day"])
    # "aug 2023 – sep 2026": lowercase English month abbreviations read as Dutch ones
    period = f"{first:%b %Y} – {last:%b %Y}".lower()
    explained = sum(stats["explained"])
    return {
        "title": "Roos en Louis groeien naar elkaars Frans toe",
        "subtitle": (
            "Roos ruilt schoolboek-Frans in voor spreektaal geleerd van Louis, Louis "
            "laat zijn Engels los.\nHun stijlprofielen lijken steeds meer op elkaar: "
            f"overeenkomst {similarity[first_q]:.2f} in {first_q[:4]}, "
            f"{similarity[last_q]:.2f} in {last_q[:4]} (1 = identiek)."
        ),
        "x_low": "← correct Frans",
        "x_low_examples": '"doit", "veux", "peut", "chez"',
        "x_high": "Engelse resten en losse leestekens →",
        "x_high_examples": '";)", " : ", " ! ", “ ” « », "or"',
        "y_low": "↓ schoolboek-Frans",
        "y_low_examples": '"beaucoup", "est-ce que", "n\'", ":)"',
        "y_high": "spreektaal ↑",
        "y_high_examples": '"ca", "faut", "oui", "quoi", " ? "',
        "footnote": (
            f"Franse WhatsApp-berichten van Roos en Louis, {period}. Trigrammen = "
            'reeksen van 3 tekens; inhoudswoorden vervangen door "x", zodat onderwerp '
            "niet meetelt.\n"
            f"Per persoon per kwartaal {settings.STYLE_DOCS_PER_QUARTER} steekproeven "
            f"van ±600 woorden → tf-idf ({stats['n_trigrams']} trigrammen) → PCA. De "
            f"twee assen tonen {explained:.0%} van de variatie.\n"
            "Toets: verschuift de stijl met de tijd? Roos omhoog naar spreektaal "
            f"ρ = {roos['rho']:.2f}, {p_short(roos['p'])} · Louis naar links "
            f"ρ = {louis['rho']:.2f}, {p_short(louis['p'])} · overeenkomst "
            f"ρ = {stats['similarity_rho']:.2f}, {p_short(stats['similarity_p'])}.\n"
            "Spearman-correlatie tussen maand en de gemiddelde positie van alle "
            f"fragmenten van 15 woorden in die maand (Roos {roos['n_months']}, Louis "
            f"{louis['n_months']} maanden met ≥{settings.STYLE_MIN_MONTH_FRAGMENTS} "
            f"fragmenten);\np-waarde via {dutch_thousands(stats['n_shuffle'])} keer de "
            "maanden door elkaar husselen. Overeenkomst = cosinus tussen de "
            f"trigramprofielen per kwartaal ({len(similarity)} kwartalen), zelfde "
            "permutatietoets."
        ),
    }


def frans_van_roos_en_louis(
    docs: pd.DataFrame, means: pd.DataFrame, stats: dict
) -> Figure:
    """Figure 8: the style map of our French, with the three tests in the footnote."""
    # The subtitle claims three directions in words; stop if a test stops saying so.
    claims = {
        "Roos towards spoken French": (
            stats["roos_trend"]["rho"],
            stats["roos_trend"]["p"],
            1,
        ),
        "Louis away from English": (
            stats["louis_trend"]["rho"],
            stats["louis_trend"]["p"],
            -1,
        ),
        "more alike": (stats["similarity_rho"], stats["similarity_p"], 1),
    }
    for claim, (rho, p, direction) in claims.items():
        if not (rho * direction > 0 and p < 0.05):
            raise ValueError(f"The subtitle says '{claim}', the test no longer does.")
    fig, _ = StyleMapPlot(PlotSettings(figsize=(12, 10), title="")).plot(
        docs=docs, means=means, texts=style_texts(stats)
    )
    return fig


def run(paths: Paths) -> None:
    """Read the stage-3 files and write all eight figures."""
    lang = pd.read_parquet(paths.language_share)
    weekly = pd.read_parquet(paths.weekly_topics)
    stats = json.loads(paths.symptom_stats.read_text())
    balance = json.loads(paths.topic_stats.read_text())
    french = pd.read_parquet(paths.weekly_french)
    french_stats = json.loads(paths.french_stats.read_text())

    save(praat_nederlands_panels(lang), "1_praat_nederlands_panels", paths)
    save(praat_nederlands_bars(lang), "2_praat_nederlands_bars", paths)
    for i, sport in [(3, "climbing"), (4, "sports")]:
        colors = {"child": settings.CHILD_BLUE, sport: settings.SPORT_ORANGE}
        fig = baby_is_taking_over(weekly, colors, balance[sport])
        save(fig, f"{i}_baby_taking_over_{sport}", paths)
    save(prepare_and_plan(stats), "5_prepare_and_plan", paths)
    save(
        familiar_language(french, french_stats),
        "6_familiar_language_fewer_words",
        paths,
    )
    home = onderweg_naar_thuis(
        pd.read_parquet(paths.home_shares), json.loads(paths.home_stats.read_text())
    )
    save(home, "7_onderweg_naar_thuis", paths)
    style = frans_van_roos_en_louis(
        pd.read_parquet(paths.style_docs),
        pd.read_parquet(paths.style_means),
        json.loads(paths.style_stats.read_text()),
    )
    save(style, "8_frans_van_roos_en_louis", paths)
    logger.success(f"figures: 8 figures -> {paths.figures}")

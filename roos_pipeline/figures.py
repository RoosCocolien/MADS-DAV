"""Stage 4 — figures: read the stage-3 tables, draw, save. No computing on messages.

One figure per lesson, the one Roos chose (2026-11-13):

    3_language_share.parquet  -> les 2  praat_nederlands_bars     (roos_02.5, code cell 5)
    3_weekly_topics.parquet   -> les 3  baby_is_taking_over       (roos_03.3, code cell 6)
    3_topic_stats.json
    3_symptom_stats.json      -> les 4  prepare_and_plan          (roos_04.6d, code cell 11)
    3_weekly_french.parquet   -> les 5  familiar_language         (roos_05.6, code cell 14)
    3_french_stats.json
    3_style_docs.parquet      -> les 6  frans_van_roos_en_louis   (roos_06.6, code cell 19)
    3_style_means.parquet
    3_style_stats.json
    3_home_shares.parquet     -> les 7  onderweg_naar_thuis       (roos_07.4, code cell 24)
    3_home_stats.json

Each figure is one function: data and settings in, `Figure` out. The only calculations
left here are presentation choices: the rolling average (how smooth a line looks) and
evaluating a fitted curve at whole numbers.

Every text on the figures is Dutch, like the report (Roos, 2026-11-13), and so are the
numbers: a decimal comma and a thousands point (`nl_number`).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from goad_toolkit.datatransforms import Pipeline, RollingAvg
from goad_toolkit.visualizer import PlotSettings, RegPlot
from loguru import logger
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter

from roos_pipeline.aggregate import nbinom_from
from roos_pipeline.plots import (
    HomeAwayPlot,
    LabelledBarPlot,
    PeakLabelledCurves,
    StyleMapPlot,
    TopicTrendPlot,
)
from roos_pipeline.results import (
    FrenchStats,
    HomeStats,
    StyleStats,
    SymptomStats,
    TopicBalanceStats,
)
from roos_pipeline.settings import (
    CATEGORIES,
    COLORS,
    GROUPING,
    LIMITS,
    TIMELINE,
    BarsPalette,
    Categories,
    FamiliarPalette,
    Grouping,
    Limits,
    Paths,
    PlanPalette,
    Timeline,
    TrendPalette,
)

# `Any`, not the inferred `float | str`: ty checks a `**` dict's value type against
# every keyword of `fig.text`, including `fontdict`, which only takes a dict.
FOOTNOTE: dict[str, Any] = {"fontsize": 8.5, "color": "dimgray"}

# The data keeps its English codes; these are the words on the figures.
LANGUAGE_NAMES = {"en": "Engels", "nl": "Nederlands", "fr": "Frans"}
PERIOD_NAMES = {"before move": "voor de verhuizing", "after move": "na de verhuizing"}
TOPIC_NAMES = {"child": "kind", "climbing": "klimmen"}
MONTHS = "jan feb mrt apr mei jun jul aug sep okt nov dec".split()


def nl_number(x: float, decimals: int = 0, sign: bool = False) -> str:
    """Dutch notation: 10.000 and 0,25; `sign` adds + or − (a real minus sign)."""
    text = f"{abs(x):,.{decimals}f}".translate(str.maketrans(",.", ".,"))
    if sign:
        return ("+" if x >= 0 else "−") + text
    return "−" + text if x < 0 else text


def nl_month(day: pd.Timestamp) -> str:
    """'aug 2023'."""
    return f"{MONTHS[day.month - 1]} {day.year}"


def save(fig: Figure, name: str, paths: Paths) -> None:
    """PNG for the report, SVG for anything that needs to scale."""
    paths.figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(paths.figures / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    logger.info(f"saved {paths.figures / name}.png/.svg")


# --- les 2: comparing categories -----------------------------------------------------


def praat_nederlands_bars(
    lang: pd.DataFrame,
    palette: BarsPalette = COLORS.bars,
    limits: Limits = LIMITS,
    categories: Categories = CATEGORIES,
) -> Figure:
    """Language mix before/after the move on one axes, before/after as hue: the 02.5
    rework after the teacher's comment that two panels mean comparing 2x3 things."""
    pct = lang.set_index(["language", "move_period"])["pct"].to_dict()
    nl_before, nl_after = pct["nl", "before move"], pct["nl", "after move"]
    fr_after = pct["fr", "after move"]
    # The subtitle says "verdrievoudigd" and "steekt schril af bij het Frans" in words:
    # if the data upstream ever change, this stops the pipeline instead of printing a
    # stale claim.
    if round(nl_after / nl_before) != 3:
        raise ValueError(
            f"Subtitle says 'verdrievoudigd', data now say {nl_after / nl_before:.1f}x"
        )
    if fr_after <= 50:
        raise ValueError("Subtitle says French still dominates: rewrite it.")

    before, after = (PERIOD_NAMES[p] for p in categories.periods)
    shown = lang.assign(
        language=lang["language"].map(LANGUAGE_NAMES),
        move_period=lang["move_period"].map(PERIOD_NAMES),
    )
    bar_settings = PlotSettings(
        figsize=(8, 5),
        title="",
        xlabel="taal",
        ylabel="% van de berichten met een herkende taal",
    )
    fig, ax = LabelledBarPlot(bar_settings).plot(
        data=shown,
        x="language",
        y="pct",
        hue="move_period",
        order=[LANGUAGE_NAMES[code] for code in categories.languages],
        hue_order=[before, after],
        palette={before: palette.before, after: palette.after},
    )
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="")
    fig.subplots_adjust(top=0.78)
    fig.suptitle("Praat Nederlands met mij", fontsize=16, fontweight="bold", y=1.0)
    fig.text(
        0.5,
        0.865,
        f"Ons Nederlands is sinds de verhuizing verdrievoudigd ({nl_before:.0f}% → "
        f"{nl_after:.0f}%), maar steekt nog schril af bij het Frans ({fr_after:.0f}%),\n"
        "terwijl Nederlands leren voor Louis belangrijker is: we wonen in Nederland.",
        ha="center",
        fontsize=10.5,
    )
    n = lang.groupby("move_period")["n"].sum()
    fig.text(
        0.5,
        -0.05,
        f"Gebaseerd op {nl_number(n['before move'])} berichten met een herkende taal "
        f"vóór de verhuizing en {nl_number(n['after move'])} erna. Berichten korter dan "
        f"{limits.min_chars} tekens tellen niet mee: te kort om iets over de taalkeuze "
        "te zeggen.",
        ha="center",
        **FOOTNOTE,
    )
    return fig


# --- les 3: time series --------------------------------------------------------------


def p_text(n_reaching: int, p: float) -> str:
    """'p ≤ 0,0001' when no shuffle reached the real value (p is then only a bound)."""
    return f"p {'≤' if n_reaching == 0 else '≈'} {nl_number(p, 4)}"


def balance_text(stats: TopicBalanceStats) -> str:
    """The shuffle test and the time-order check, as two sentences under the chart."""
    sport = TOPIC_NAMES.get(stats.sport, stats.sport)
    before = nl_number(100 * stats.mean_gap_before, 1, sign=True)
    after = nl_number(100 * stats.mean_gap_after, 1, sign=True)
    first = (
        f"Kind min {sport}, in procentpunten van de berichten per week: {before} vóór "
        f"de positieve test, {after} erna ({stats.n_before} en {stats.n_after} weken; "
        "de week van de test telt niet mee).\nToen we het label vóór/na "
        f"{nl_number(stats.n_shuffle)} keer willekeurig over alle weken husselden, kwam "
        f"een verschuiving van deze omvang {stats.n_shuffles_reaching_shift} keer voor "
        f"({p_text(stats.n_shuffles_reaching_shift, stats.p_value_greater)})."
    )
    if stats.max_run_gap is None:
        second = (
            f"Te weinig weken vóór de test voor een reeks van {stats.run_weeks} "
            "opeenvolgende weken: geen controle op de volgorde in de tijd."
        )
    else:
        second = (
            f"Met de volgorde in de tijd intact: {stats.n_runs_reaching_after} van de "
            f"{stats.n_runs} reeksen van {stats.run_weeks} opeenvolgende weken vóór de "
            "test haalden het gemiddelde van na de test (hoogste reeks "
            f"{nl_number(100 * stats.max_run_gap, 1, sign=True)})."
        )
    return f"{first}\n{second}"


def baby_is_taking_over(
    weekly: pd.DataFrame,
    balance: TopicBalanceStats,
    palette: TrendPalette = COLORS.trend,
    grouping: Grouping = GROUPING,
    timeline: Timeline = TIMELINE,
) -> Figure:
    """Weekly share of messages about the child and about climbing, with the test and
    the birth marked, and the statistical test (computed in aggregate.py) underneath."""
    colors = {"child": palette.child, "climbing": palette.climbing}
    trend = Pipeline()
    for topic in colors:
        trend.add(RollingAvg, column=topic, window=grouping.smooth_weeks, rename=True)
    weekly_trend = trend.apply(weekly)

    trend_settings = PlotSettings(
        figsize=(12, 5),
        title="De baby neemt het over",
        xlabel="",
        ylabel="aandeel van de berichten per week",
        xtick_rotation=45,
    )
    fig, ax = TopicTrendPlot(trend_settings).plot(
        weekly=weekly,
        trend=weekly_trend,
        x="week",
        colors=colors,
        labels={
            t: f"{TOPIC_NAMES[t]} (gemiddelde over {grouping.smooth_weeks} weken)"
            for t in colors
        },
        events={
            "positieve zwangerschapstest": str(timeline.test.date()),
            "Benji is geboren": str(timeline.birth.date()),
        },
    )
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    # Every weekly share is a share of that week's messages: say how many that is.
    n = weekly["n_messages"]
    mentions = " · ".join(
        f"{TOPIC_NAMES[t]} {nl_number(int(weekly[f'{t}_n'].sum()))}" for t in colors
    )
    fig.text(
        0.5,
        -0.08,
        f"Gebaseerd op {nl_number(int(n.sum()))} berichten over {len(weekly)} weken "
        f"(mediaan {int(n.median())} per week, minimaal {int(n.min())}). "
        f"Berichten per onderwerp: {mentions}.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.text(0.5, -0.12, balance_text(balance), ha="center", va="top", **FOOTNOTE)
    return fig


# --- les 4: distributions ------------------------------------------------------------


def prepare_and_plan(stats: SymptomStats, palette: PlanPalette = COLORS.plan) -> Figure:
    """Normal weeks vs. hard weeks as two fitted negative binomials, shuffle test below."""
    ratio = stats.mean_hard / stats.mean_normal
    # The subtitle says "drie keer" in words: if the data upstream ever changes, this
    # stops the pipeline instead of printing a claim the numbers no longer make.
    if round(ratio) != 3:
        raise ValueError(f"Subtitle says 'drie keer', data now say {ratio:.1f}x")

    curves = pd.DataFrame({"k": range(stats.max_count + 3)})
    curves["normal"] = nbinom_from(stats.normal_mu, stats.normal_r).pmf(curves["k"])
    curves["hard"] = nbinom_from(stats.hard_mu, stats.hard_r).pmf(curves["k"])

    both_settings = PlotSettings(
        figsize=(10, 6),
        title="",
        xlabel="berichten over misselijkheid of vermoeidheid, per week",
        ylabel="kans: aandeel van de weken met precies zoveel",
    )
    fig, ax = PeakLabelledCurves(both_settings).plot(
        data=curves,
        x="k",
        curves={
            "normal": (palette.grey, "normale weken"),
            "hard": (palette.hard, "zware weken"),
        },
    )
    ax.set_xlim(-0.5, curves["k"].max() + 0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: nl_number(y, 2)))
    ax.set_title(
        "In de zware weken ging het drie keer zo vaak over misselijkheid en "
        f"vermoeidheid: {stats.mean_hard:.0f} berichten per week in plaats van "
        f"{stats.mean_normal:.0f}",
        loc="left",
        fontsize=12,
        color="dimgray",
    )
    fig.suptitle(
        "Hoe overleef je een zwangerschap: voorbereiden en plannen.",
        x=0.125,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.02,
        f"Grijs: {stats.n_normal} normale weken op afstand, vóór de zwangerschap "
        f"(gemiddeld {nl_number(stats.mean_normal, 1)} per week). Rood: de "
        f"{stats.n_hard} zware weken, zwangerschapsweek 5-14 (gemiddeld "
        f"{nl_number(stats.mean_hard, 1)}).\n"
        "Beide lijnen: een gefitte negatief-binomiale verdeling.",
        va="top",
        **FOOTNOTE,
    )
    fig.text(
        0.01,
        -0.08,
        "Toeval alleen verklaart dit verschil niet: toen het label 'zwaar' "
        f"{nl_number(stats.n_shuffle)} keer willekeurig over alle "
        f"{stats.n_weeks_shuffled} weken werd gehusseld,\nkwam een verschil zo groot "
        f"als het echte ({nl_number(stats.gap_observed, 1, sign=True)} "
        f"symptoomberichten per week) {stats.n_shuffles_reaching_gap} keer voor "
        f"(p ≤ {nl_number(stats.p_value_greater, 4)}).",
        va="top",
        **FOOTNOTE,
    )
    return fig


# --- les 5: relationships ------------------------------------------------------------


def language_caption(stats: FrenchStats, limits: Limits = LIMITS) -> str:
    """How many of Louis's messages the dots rest on, and their language mix.

    Percentages are over the classified messages, so they add up to 100%; the
    "unknown" ones are counted separately.
    """
    counts = stats.language_counts
    known = sum(counts[code] for code in LANGUAGE_NAMES)
    shares = ", ".join(
        f"{100 * counts[code] / known:.0f}% {name}"
        for code, name in LANGUAGE_NAMES.items()
    )
    return (
        f"Gebaseerd op {nl_number(stats.n_messages)} berichten van Louis over "
        f"{stats.n_weeks} weken: {shares}\n"
        f"(van de {nl_number(known)} berichten die lang genoeg zijn om een taal te "
        f"herkennen; {nl_number(counts['unknown'])} berichten korter dan "
        f"{limits.min_chars} tekens hebben geen taal, maar tellen wel mee voor de "
        "woorden per bericht)."
    )


def familiar_language(
    weekly: pd.DataFrame,
    stats: FrenchStats,
    palette: FamiliarPalette = COLORS.familiar,
    limits: Limits = LIMITS,
) -> Figure:
    """One dot per long-distance week: Louis's French share against his median message
    length, with the log-of-x line and the within-week test underneath."""
    # The footnote says "p < 0,001" and "korter" in words: if the data upstream ever
    # change, this stops the pipeline instead of printing a claim the test no longer makes.
    if not (stats.wilcoxon_p_less < 0.001 and stats.median_difference < 0):
        raise ValueError(
            "The test no longer supports the line under the graph: rewrite it."
        )

    final = PlotSettings(
        figsize=(9, 6),
        title="",
        xlabel="aandeel Franse berichten die week",
        ylabel="woorden per bericht (mediaan van die week)",
    )
    # logx: median words = a + b * ln(share), the line chosen in roos_05.6 §5.6.5.
    fig, ax = RegPlot(final).plot(
        data=weekly,
        x="french_share",
        y="median_words",
        logx=True,
        ci=None,
        color=palette.dark_grey,
        scatter_kws={"color": palette.dots, "alpha": 0.75},
    )
    ax.set(xlabel=final.xlabel, ylabel=final.ylabel)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: nl_number(y, 1)))
    ax.text(
        0.98,
        0.80,
        "lijn: elke verdubbeling van het Frans\n"
        f"= {nl_number(abs(stats.words_per_doubling), 1)} woorden minder per bericht",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=palette.dark_grey,
    )

    fig.subplots_adjust(top=0.84)
    fig.suptitle(
        "Vertrouwde taal heeft minder woorden nodig",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.905,
        "Toen we nog niet samenwoonden, schreef Louis kortere berichten als hij in het "
        "Frans schreef.",
        ha="center",
        fontsize=11,
    )

    first_day = pd.Timestamp(stats.first_day)
    last_week = pd.Timestamp(stats.last_week)
    footnote = "\n".join(
        [
            "Elke stip is een week dat we nog niet samenwoonden "
            f"({nl_month(first_day)} – {nl_month(last_week)}): welk deel van Louis' "
            "berichten die week Frans was, en hoeveel woorden zijn middelste bericht had.",
            language_caption(stats, limits),
            "In dezelfde week waren zijn Franse berichten korter dan zijn Engelse en "
            f"Nederlandse in {stats.n_weeks_below} van de {stats.n_weeks_compared} weken "
            f"(meestal {abs(stats.median_difference):.0f} woorden minder; "
            "Wilcoxon-rangtekentoets, p < 0,001).",
        ]
    )
    fig.text(0.5, -0.02, footnote, ha="center", va="top", linespacing=1.5, **FOOTNOTE)
    return fig


# --- les 6: Roos en Louis groeien naar elkaars Frans toe -----------------------------


def p_short(p: float) -> str:
    return "p < 0,001" if p < 0.001 else f"p = {nl_number(p, 3)}"


def style_texts(
    stats: StyleStats, grouping: Grouping = GROUPING, limits: Limits = LIMITS
) -> dict[str, str]:
    """Every sentence on the les-6 figure. The axis examples come from the loadings (06.6)."""
    similarity = stats.similarity
    first_q, last_q = next(iter(similarity)), list(similarity)[-1]
    roos, louis = stats.roos_trend, stats.louis_trend
    first, last = pd.Timestamp(stats.first_day), pd.Timestamp(stats.last_day)
    explained = sum(stats.explained)
    return {
        "title": "Roos en Louis groeien naar elkaars Frans toe",
        "subtitle": (
            "Roos ruilt schoolboek-Frans in voor spreektaal geleerd van Louis, Louis "
            "laat zijn Engels los.\nHun stijlprofielen lijken steeds meer op elkaar: "
            f"overeenkomst {nl_number(similarity[first_q], 2)} in {first_q[:4]}, "
            f"{nl_number(similarity[last_q], 2)} in {last_q[:4]} (1 = identiek)."
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
            f"Franse WhatsApp-berichten van Roos en Louis, {nl_month(first)} – "
            f"{nl_month(last)}. Trigrammen = reeksen van 3 tekens; inhoudswoorden "
            'vervangen door "x", zodat onderwerp niet meetelt.\n'
            f"Per persoon per kwartaal {grouping.docs_per_quarter} steekproeven "
            f"van ±600 woorden → tf-idf ({stats.n_trigrams} trigrammen) → PCA. De "
            f"twee assen tonen {explained:.0%} van de variatie.\n"
            "Toets: verschuift de stijl met de tijd? Roos omhoog naar spreektaal "
            f"ρ = {nl_number(roos.rho, 2)}, {p_short(roos.p)} · Louis naar links "
            f"ρ = {nl_number(louis.rho, 2)}, {p_short(louis.p)} · overeenkomst "
            f"ρ = {nl_number(stats.similarity_rho, 2)}, {p_short(stats.similarity_p)}.\n"
            "Spearman-correlatie tussen maand en de gemiddelde positie van alle "
            f"fragmenten van {grouping.fragment_words} woorden in die maand (Roos "
            f"{roos.n_months}, Louis {louis.n_months} maanden met "
            f"≥{limits.min_month_fragments} fragmenten);\np-waarde via "
            f"{nl_number(stats.n_shuffle)} keer de maanden door elkaar husselen. "
            "Overeenkomst = cosinus tussen de trigramprofielen per kwartaal "
            f"({len(similarity)} kwartalen), zelfde permutatietoets."
        ),
    }


def frans_van_roos_en_louis(
    docs: pd.DataFrame, means: pd.DataFrame, stats: StyleStats
) -> Figure:
    """Les 6: the style map of our French, with the three tests in the footnote."""
    # The subtitle claims three directions in words; stop if a test stops saying so.
    claims = {
        "Roos towards spoken French": (stats.roos_trend.rho, stats.roos_trend.p, 1),
        "Louis away from English": (stats.louis_trend.rho, stats.louis_trend.p, -1),
        "more alike": (stats.similarity_rho, stats.similarity_p, 1),
    }
    for claim, (rho, p, direction) in claims.items():
        if not (rho * direction > 0 and p < 0.05):
            raise ValueError(f"The subtitle says '{claim}', the test no longer does.")
    fig, _ = StyleMapPlot(PlotSettings(figsize=(12, 10), title="")).plot(
        docs=docs, means=means, texts=style_texts(stats)
    )
    return fig


# --- les 7: Van onderweg naar thuis --------------------------------------------------


def home_caption(
    stats: HomeStats, grouping: Grouping = GROUPING, limits: Limits = LIMITS
) -> str:
    """The three paragraphs under the les-7 figure: dots and lines, themes, the test."""
    by_message, by_day = stats.messages, stats.days
    n = nl_number(stats.n_shuffle)
    p_floor = nl_number(1 / (stats.n_shuffle + 1), 4)
    against = stats.against
    return (
        "Elke donkere stip is een stad, gerangschikt op thuis-score: hoe verder naar "
        "rechts, hoe meer haar gesprekken over thuis met kind gaan in plaats van "
        "spontaan op pad. Dit is een bipartiet (tweeledig) netwerk: een lijn loopt "
        "alleen tussen een stad en een thema, nooit stad-stad of thema-thema. De "
        "dikte is hoe vaak het thema voorkomt in een gesprekje over de stad (het "
        f"bericht dat de stad noemt plus de {grouping.home_window} erna); het "
        "percentage bij Parijs en Nieuwegein is dat aandeel zelf -- hoeveel procent "
        "van de berichten die de stad noemen zo'n gesprekje opleveren. Lijnen onder "
        f"{limits.home_min_share:.0%} zijn weggelaten.\n\n"
        "De zes thema's vallen in twee groepen: spontaan op pad (reizen, klimmen & "
        "sport, eten & uitgaan) en thuis met kind (Benji & kind, wonen, familie), "
        "gevonden met woordenlijsten in het Nederlands, Frans en Engels (trein/TGV, "
        "klimmen/escalade, restaurant/apéro, Benji/crèche, huis/verhuizen, "
        "ouders/oma) -- een woord kan ook iets anders betekenen.\n\n"
        f"Schudtest. De thuis-score van {stats.place} ligt "
        f"{by_message.gap_observed * 100:.0f} procentpunt hoger dan die van "
        f"{', '.join(against[:-1])} en {against[-1]} samen. {n} keer willekeurig "
        "gehusseld welke stad bij welk gesprekje hoort gaf "
        f"{by_message.n_shuffles_reaching_gap} keer zo'n verschil (p < {p_floor}); "
        f"per dag in plaats van per bericht {by_day.n_shuffles_reaching_gap} keer. "
        "Vrijwel zeker geen toeval; waarom het er is, zegt de test niet."
    )


def onderweg_naar_thuis(shares: pd.DataFrame, stats: HomeStats) -> Figure:
    """Les 7: the places between the on-the-road and the home themes."""
    # "vrijwel zeker geen toeval" in words: stop if the test stops saying so.
    for unit, result in (("messages", stats.messages), ("days", stats.days)):
        if not (result.gap_observed > 0 and result.p_value_greater < 0.01):
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


def run(paths: Paths) -> None:
    """Read the stage-3 files and write the six figures, one per lesson."""
    lang = pd.read_parquet(paths.language_share)
    weekly = pd.read_parquet(paths.weekly_topics)
    balance = TopicBalanceStats.model_validate_json(paths.topic_stats.read_text())
    symptoms = SymptomStats.model_validate_json(paths.symptom_stats.read_text())
    french = pd.read_parquet(paths.weekly_french)
    french_stats = FrenchStats.model_validate_json(paths.french_stats.read_text())
    style_stats = StyleStats.model_validate_json(paths.style_stats.read_text())
    home_stats = HomeStats.model_validate_json(paths.home_stats.read_text())

    save(praat_nederlands_bars(lang), "les_2_praat_nederlands_bars", paths)
    save(
        baby_is_taking_over(weekly, balance),
        "les_3_baby_taking_over_climbing_time",
        paths,
    )
    save(prepare_and_plan(symptoms), "les_4_prepare_and_plan_distributie", paths)
    save(
        familiar_language(french, french_stats),
        "les_5_fam_lang_fewer_words_relation",
        paths,
    )
    style = frans_van_roos_en_louis(
        pd.read_parquet(paths.style_docs),
        pd.read_parquet(paths.style_means),
        style_stats,
    )
    save(style, "les_6_frans_roos_louis_high_dim", paths)
    home = onderweg_naar_thuis(pd.read_parquet(paths.home_shares), home_stats)
    save(home, "les_7_onderweg_naar_thuis_netwerk", paths)
    logger.success(f"figures: 6 figures -> {paths.figures}")

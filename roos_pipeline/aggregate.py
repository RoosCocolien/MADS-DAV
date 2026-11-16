"""Stage 3 — aggregate: change the grain from messages to periods and weeks, and fit.

    2_features.parquet
      -> language_share        % per language per move period  -> 3_language_share.parquet
      -> weekly_topics         topic share per calendar week    -> 3_weekly_topics.parquet
           -> topic_balance_stats   child minus climbing, before vs after the test:
                                    shuffle test + runs of 10 weeks -> 3_topic_stats.json
      -> weekly_symptoms       symptom messages per pregnancy week
                                                                -> 3_weekly_symptoms.parquet
           -> symptom_stats    negative binomial fits + shuffle test
                                                                -> 3_symptom_stats.json
      -> louis_messages -> weekly_french
                               French share and words per message per long-distance
                               week                             -> 3_weekly_french.parquet
           -> french_length_stats  log fit + within-week Wilcoxon test
                                                                -> 3_french_stats.json

    2_unmerged_features.parquet
      -> theme_shares          share per (place, theme), thuis-score
                                                                -> 3_home_shares.parquet
      -> place_talks -> home_stats
                               Nieuwegein vs. Parijs/Amsterdam/Utrecht: shuffle test,
                               per message and per day          -> 3_home_stats.json
      -> style_map             French units -> masked fragments -> documents ->
                               tf-idf + PCA(2); trends + similarity with shuffle p
                                                                -> 3_style_docs.parquet
                                                                -> 3_style_means.parquet
                                                                -> 3_style_stats.json

Everything after this stage works on a few hundred rows instead of 63,000 messages,
and the slowest step of the whole pipeline (10,000 shuffles) runs here, once. The
figures stage only reads these small files, so a title can change without
re-running any of it.
"""

from __future__ import annotations

import re
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from goad_toolkit.analytics import NullDistribution
from goad_toolkit.datatransforms import Filter, Pipeline
from loguru import logger
from scipy import optimize, stats
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from roos_pipeline.features import PregnancyWeek
from roos_pipeline.results import (
    FrenchStats,
    HomeStats,
    HomeTestResult,
    StyleStats,
    SymptomStats,
    TopicBalanceStats,
    Trend,
)
from roos_pipeline.settings import (
    CATEGORIES,
    GROUPING,
    LIMITS,
    PATTERNS,
    TESTS,
    TIMELINE,
    WORD_LISTS,
    Categories,
    Grouping,
    HomeTest,
    Limits,
    Paths,
    Patterns,
    ShuffleTest,
    Side,
    Tests,
    Timeline,
    TopicBalanceTest,
    WordLists,
)
from scripts.sessionize import merge_messages, sessionize

if TYPE_CHECKING:
    # scipy has no public name for a frozen distribution; only needed for the type.
    from scipy.stats._distn_infrastructure import rv_discrete_frozen

SYMPTOMS = ["nausea", "fatigue", "either"]
NORMAL, HARD = "before pregnancy", "hard weeks"


# --- Praat Nederlands met mij ------------------------------------------------------


def language_share(features: pd.DataFrame) -> pd.DataFrame:
    """Percentage of each language among *classified* messages, per move period.

    "unknown" (too short to classify) is left out of the denominator, so the bars
    within one period add up to 100%. `n` is kept so the chart can say what the
    percentages rest on.
    """
    known = features[features["language"] != "unknown"]
    counts = known.groupby(["move_period", "language"]).size().rename("n").reset_index()
    counts["pct"] = counts.groupby("move_period")["n"].transform(
        lambda s: 100 * s / s.sum()
    )
    return counts


# --- Baby is taking over -----------------------------------------------------------


def weekly_topics(
    features: pd.DataFrame, categories: Categories = CATEGORIES
) -> pd.DataFrame:
    """Per calendar week: each topic's share of that week's messages, and the counts.

    A share, not a count (03.3.2's rule): message volume itself halves after the move,
    and a count would re-tell that drop instead of a change in topic. A week without
    messages keeps a NaN share, so the chart breaks the line instead of inventing a
    zero.
    """
    by_week = features.set_index("ts_local")[list(categories.trend_topics)].resample(
        "W"
    )
    shares = by_week.mean()
    counts = by_week.sum().add_suffix("_n")
    n_messages = by_week.size().rename("n_messages")
    return (
        pd.concat([shares, counts, n_messages], axis=1)
        .rename_axis("week")
        .reset_index()
    )


BEFORE_TEST, AFTER_TEST = "before test", "after test"


def mean_shift(frame: pd.DataFrame) -> float:
    """Weekly gap (child share minus sport share): after-test average minus before."""
    means = frame.groupby("period")["gap"].mean()
    return means[AFTER_TEST] - means[BEFORE_TEST]


def topic_balance_stats(
    weekly: pd.DataFrame,
    sport: str,
    test: TopicBalanceTest = TESTS.topic_balance,
    timeline: Timeline = TIMELINE,
) -> TopicBalanceStats:
    """Did the balance between child and sport tip at the positive test, beyond chance?

    One number per week: child's share minus `sport`'s share of that week's messages
    (Roos's choice, 2026-10-06: one test on the gap, "the balance tipped", rather than
    two separate tests). Weeks before the test against weeks from the test on; the one
    week that contains the test is left out, because it is half of each.

    Two checks, the same pair as roos_04.6d:

    1. **Shuffle** the before/after label over all weeks (`NullDistribution`): how
       often does a random split produce a shift this large? This treats weeks as
       interchangeable, which neighbouring weeks are not quite.
    2. **Keep the order of time:** every run of `test.run_weeks` consecutive weeks before
       the test. Did the time before ever produce a stretch with a gap as high as the
       after-test average, on its own?

    `resample("W")` labels each week by its last day (a Sunday), so a week starts six
    days before its label.
    """
    week_end = weekly["week"]
    week_start = week_end - pd.Timedelta(days=6)
    frame = pd.DataFrame(
        {
            "gap": weekly["child"] - weekly[sport],
            "period": np.select(
                [week_end < timeline.test, week_start >= timeline.test],
                [BEFORE_TEST, AFTER_TEST],
                default="",
            ),
        }
    )
    frame = frame[(frame["period"] != "") & frame["gap"].notna()]
    before = frame.loc[frame["period"] == BEFORE_TEST, "gap"]
    after = frame.loc[frame["period"] == AFTER_TEST, "gap"]

    null = NullDistribution(mean_shift, n_iter=test.n_shuffle, seed=test.seed).run(
        frame, label="period"
    )
    # Rolling over the before-weeks in calendar order; with fewer weeks than one run
    # there is no run at all, and the result says so instead of inventing one.
    runs = before.reset_index(drop=True).rolling(test.run_weeks).mean().dropna()
    return TopicBalanceStats(
        sport=sport,
        n_before=len(before),
        n_after=len(after),
        mean_gap_before=float(before.mean()),
        mean_gap_after=float(after.mean()),
        shift_observed=float(null.observed),
        n_shuffle=test.n_shuffle,
        n_shuffles_reaching_shift=int((null.null >= null.observed).sum()),
        p_value_greater=float(null.p_value("greater")),
        run_weeks=test.run_weeks,
        n_runs=len(runs),
        max_run_gap=float(runs.max()) if len(runs) else None,
        n_runs_reaching_after=int((runs >= after.mean()).sum()),
    )


# --- How to survive a pregnancy: prepare and plan ----------------------------------


def daily_symptoms(features: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day, days without messages included as zeros."""
    day = features["ts_local"].dt.normalize()
    daily = (
        features.groupby(day)[SYMPTOMS]
        .sum()
        # value_counts on the day Series, not DataFrame.groupby().size(): pandas types
        # the latter as "Series or DataFrame", and ty (rightly) won't call .rename on
        # a maybe-DataFrame. Same numbers either way.
        .join(day.value_counts().rename("messages"))
    )
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_range, fill_value=0).rename_axis("day").reset_index()


def weekly_symptoms(
    features: pd.DataFrame, timeline: Timeline = TIMELINE
) -> pd.DataFrame:
    """Symptom messages per pregnancy week, every week a full 7 days.

    Weeks are pregnancy weeks (see `PregnancyWeek`), so the hard period is exactly
    weeks 5-14. The partial first and last block are dropped: a 3-day "week" would
    look like a quiet week. `group` labels the two sides of the comparison.
    """
    daily = Pipeline().add(PregnancyWeek, column="day", start=timeline.pregnancy_start)
    days = daily.apply(daily_symptoms(features))
    weekly = days.groupby("pregnancy_week").agg(
        start=("day", "min"),
        n_days=("day", "size"),
        messages=("messages", "sum"),
        **{s: (s, "sum") for s in SYMPTOMS},
    )
    weekly = weekly[weekly["n_days"] == 7].copy()
    weekly["share"] = (
        1000 * weekly["either"] / weekly["messages"].where(weekly["messages"] > 0)
    )
    weekly["group"] = np.select(
        [
            weekly.index <= timeline.last_normal_week,
            weekly.index.isin(timeline.hard_weeks),
        ],
        [NORMAL, HARD],
        default="other",
    )
    return weekly.reset_index()


def nbinom_from(mu: float, r: float) -> rv_discrete_frozen:
    """scipy's nbinom takes (n, p); n is r, and p = r / (r + mu).

    Calling `stats.nbinom` (an `nbinom_gen`) with both parameters fixes them, and
    gives back a *frozen* distribution with .pmf/.logpmf ready to use.
    """
    return stats.nbinom(r, r / (r + mu))


def aic(log_likelihood: float, k: int) -> float:
    return 2 * k - 2 * log_likelihood


def fit_daily(values: np.ndarray) -> dict:
    """Poisson and negative binomial by maximum likelihood, compared on AIC.

    Named for its first use on daily counts (roos_04.6c §4.6.12 step 1, which also
    explains why this isn't goad's `DistributionFitter`); here it runs on weekly ones.
    """
    poisson_mu = values.mean()  # the Poisson MLE is the sample mean
    poisson_ll = stats.poisson.logpmf(values, poisson_mu).sum()

    def nll(theta: np.ndarray) -> float:  # log-parameters keep mu and r positive
        return -nbinom_from(np.exp(theta[0]), np.exp(theta[1])).logpmf(values).sum()

    res = optimize.minimize(
        nll,
        [np.log(values.mean()), 0.0],
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8},
    )
    mu, r = np.exp(res.x)
    return {
        "n_days": len(values),
        "mean": values.mean(),
        "vmr": values.var(ddof=1) / values.mean(),
        "poisson_mu": poisson_mu,
        "poisson_aic": aic(poisson_ll, 1),
        "nb_mu": mu,
        "nb_r": r,
        "nb_alpha": 1 / r,
        "nb_aic": aic(-res.fun, 2),
        "nb_dist": nbinom_from(mu, r),
    }


def mean_gap(frame: pd.DataFrame) -> float:
    """Symptom messages per week: hard-weeks average minus normal-weeks average."""
    means = frame.groupby("group")["either"].mean()
    return means[HARD] - means[NORMAL]


def symptom_stats(
    weekly: pd.DataFrame,
    test: ShuffleTest = TESTS.symptoms,
    timeline: Timeline = TIMELINE,
) -> SymptomStats:
    """Fit both groups, and shuffle the group label to see what chance produces.

    Returns plain numbers only (no scipy objects), so the result can be written to
    JSON and the figure stage can rebuild each curve from `mu` and `r`.
    """
    two_groups = weekly[weekly["group"].isin([NORMAL, HARD])]
    normal = two_groups.loc[two_groups["group"] == NORMAL, "either"].to_numpy(float)
    hard = two_groups.loc[two_groups["group"] == HARD, "either"].to_numpy(float)
    if len(hard) != len(timeline.hard_weeks):
        raise ValueError(
            f"Expected {len(timeline.hard_weeks)} full hard weeks, got {len(hard)}"
        )

    normal_fit, hard_fit = fit_daily(normal), fit_daily(hard)
    null = NullDistribution(mean_gap, n_iter=test.n_shuffle, seed=test.seed).run(
        two_groups[["either", "group"]], label="group"
    )
    return SymptomStats(
        n_normal=len(normal),
        n_hard=len(hard),
        mean_normal=float(normal.mean()),
        mean_hard=float(hard.mean()),
        max_count=int(max(normal.max(), hard.max())),
        normal_mu=float(normal_fit["nb_mu"]),
        normal_r=float(normal_fit["nb_r"]),
        hard_mu=float(hard_fit["nb_mu"]),
        hard_r=float(hard_fit["nb_r"]),
        hard_delta_aic_vs_poisson=float(hard_fit["poisson_aic"] - hard_fit["nb_aic"]),
        n_weeks_shuffled=len(two_groups),
        n_shuffle=test.n_shuffle,
        gap_observed=float(null.observed),
        n_shuffles_reaching_gap=int((null.null >= null.observed).sum()),
        p_value_greater=float(null.p_value("greater")),
    )


# --- Familiar language needs fewer words -------------------------------------------


def louis_messages(features: pd.DataFrame) -> pd.DataFrame:
    """Louis's messages, each with the calendar week it belongs to.

    The week is labelled by its Sunday, the same label `resample("W")` gives, so a
    message can be matched to its row in `weekly_french`.
    """
    louis = Pipeline().add(Filter, expr="author == 'Louis'").apply(features)
    louis["week"] = louis["ts_local"].dt.to_period("W-SUN").dt.end_time.dt.normalize()
    return louis


def weekly_french(
    louis: pd.DataFrame, limits: Limits = LIMITS, timeline: Timeline = TIMELINE
) -> pd.DataFrame:
    """Per calendar week while we lived apart: Louis's French share and message length.

    The French share is over his classified messages only (`is_french` is NaN for
    "unknown"); words per message count all of them, the short ones included. Only
    weeks that end before the move (the week of the move is half of each), and with
    at least `limits.min_messages` messages: one long message makes a week of 3 look
    long.
    """
    by_week = louis.set_index("ts_local").resample("W")
    weekly = pd.DataFrame(
        {
            "french_share": by_week["is_french"].mean(),
            "words_per_message": by_week["n_words"].mean(),
            "median_words": by_week["n_words"].median(),
            "n_messages": by_week.size(),
        }
    ).rename_axis("week")
    weekly = weekly.reset_index()
    keep = (weekly["week"] < timeline.move) & (
        weekly["n_messages"] >= limits.min_messages
    )
    return weekly[keep].dropna(subset=["french_share"]).reset_index(drop=True)


def within_week_differences(
    messages: pd.DataFrame, limits: Limits = LIMITS
) -> pd.Series:
    """Per week: median words of his French messages minus median of his other ones.

    A week needs `limits.min_per_group` classified messages in each group, otherwise one
    message would be "the median". Below 0: his French messages were shorter that week.
    """
    classified = messages[messages["is_french"].notna()]
    group = np.where(classified["is_french"] == 1.0, "French", "other")
    per_week = (
        classified.groupby(["week", group])["n_words"].agg(["median", "size"]).unstack()
    )
    both = per_week[
        (per_week[("size", "French")] >= limits.min_per_group)
        & (per_week[("size", "other")] >= limits.min_per_group)
    ]
    return (both[("median", "French")] - both[("median", "other")]).rename("difference")


def french_length_stats(
    louis: pd.DataFrame,
    weekly: pd.DataFrame,
    limits: Limits = LIMITS,
    categories: Categories = CATEGORIES,
) -> FrenchStats:
    """The fitted line and the within-week test, as plain numbers for the figure.

    The line: median words = a + b * ln(French share), so every doubling of the share
    changes a week's median message by b * ln(2) words (roos_05.6 §5.6.7).

    The test (roos_05.6 §5.6.6): within each week, are his French messages shorter
    than his English and Dutch ones? Everything that is the same within one week
    (long distance, pregnant, how busy) cannot make that difference. Wilcoxon
    signed-rank on one difference per week, one-sided because the hypothesis has a
    direction. Neighbouring weeks are not fully independent, so p is too optimistic.
    """
    apart = louis[louis["week"].isin(weekly["week"])]
    diff = within_week_differences(apart, limits)
    wilcoxon = stats.wilcoxon(diff, alternative="less")
    b, a = np.polyfit(np.log(weekly["french_share"]), weekly["median_words"], 1)
    counts = apart["language"].value_counts()
    return FrenchStats(
        n_weeks=len(weekly),
        first_day=str((weekly["week"].min() - pd.Timedelta(days=6)).date()),
        last_week=str(weekly["week"].max().date()),
        n_messages=len(apart),
        language_counts={
            lang: int(counts.get(lang, 0))
            for lang in [*categories.languages, "unknown"]
        },
        fit_a=float(a),
        fit_b=float(b),
        words_per_doubling=float(b * np.log(2)),
        n_weeks_compared=len(diff),
        n_weeks_below=int((diff < 0).sum()),
        median_difference=float(diff.median()),
        wilcoxon_p_less=float(wilcoxon.pvalue),
    )


# --- Van onderweg naar thuis (roos_07.4 §7.4.13) ------------------------------------


def theme_shares(features: pd.DataFrame, patterns: Patterns = PATTERNS) -> pd.DataFrame:
    """One row per place: `berichten`, one share column per theme, and `thuis_score`.

    share(place, theme) = messages naming the place whose gesprekje holds the theme /
    messages naming the place. thuis_score = the summed shares of the home themes /
    the summed shares of all themes.
    """
    themes = patterns.home_themes
    rows = {}
    for place in patterns.home_places:
        here = features[f"place_{place}"]
        n = int(here.sum())
        rows[place] = {"berichten": n} | {
            theme: float((features[f"talk_{theme}"] & here).sum() / n) if n else 0.0
            for theme in themes
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    home = [t for t, theme in themes.items() if theme.group == Side.HOME]
    table["thuis_score"] = table[home].sum(axis=1) / table[list(themes)].sum(axis=1)
    return table.rename_axis("place")


def place_talks(features: pd.DataFrame, patterns: Patterns = PATTERNS) -> pd.DataFrame:
    """One row per message naming exactly one place: `day`, `place`, `home`, `away`.

    `home` and `away` count the themes of that group in the message's gesprekje.
    Summed over a place they give its thuis-score (home / (home + away)), because
    every share has the same denominator. A message naming two places is left out:
    it has no single label to shuffle.
    """
    places = list(patterns.home_places)
    named = features[[f"place_{p}" for p in places]].to_numpy(dtype=bool)
    only_one = named.sum(axis=1) == 1
    count = {
        side: sum(
            features[f"talk_{t}"].to_numpy(dtype=int)
            for t, theme in patterns.home_themes.items()
            if theme.group == side
        )
        for side in Side
    }
    return pd.DataFrame(
        {
            "day": features["ts_local"].dt.normalize().to_numpy()[only_one],
            "place": np.array(places)[named[only_one].argmax(axis=1)],
            "home": np.asarray(count[Side.HOME])[only_one],
            "away": np.asarray(count[Side.AWAY])[only_one],
        }
    )


def home_gap(talks: pd.DataFrame, test: HomeTest = TESTS.home) -> float:
    """thuis-score of `test.place` minus that of `test.against` taken together."""

    def score(rows: pd.DataFrame) -> float:
        return rows["home"].sum() / (rows["home"].sum() + rows["away"].sum())

    return score(talks[talks["place"] == test.place]) - score(
        talks[talks["place"].isin(test.against)]
    )


def home_stats(
    features: pd.DataFrame, test: HomeTest = TESTS.home, patterns: Patterns = PATTERNS
) -> HomeStats:
    """The shuffle test of the les-7 figure, with a message and with a day as the unit.

    Shuffle the place label 10,000 times: how often does chance give a gap in
    thuis-score as large as the real one? Messages on one day are no independent
    draws (one evening about a house is ten messages), so the same test runs once
    more on one row per (day, place), with that day's themes summed. One-sided:
    the claim is that the place is *more* about home.
    """
    talks = place_talks(features, patterns)
    per_day = talks.groupby(["day", "place"])[["home", "away"]].sum().reset_index()
    gap = partial(home_gap, test=test)
    results = {}
    for unit, frame in (("messages", talks), ("days", per_day)):
        null = NullDistribution(gap, n_iter=test.n_shuffle, seed=test.seed).run(
            frame, label="place"
        )
        results[unit] = HomeTestResult(
            n_place=int((frame["place"] == test.place).sum()),
            n_against=int(frame["place"].isin(test.against).sum()),
            gap_observed=float(null.observed),
            n_shuffles_reaching_gap=int((null.null >= null.observed).sum()),
            p_value_greater=float(null.p_value("greater")),
        )
    return HomeStats(
        place=test.place,
        against=list(test.against),
        n_shuffle=test.n_shuffle,
        messages=results["messages"],
        days=results["days"],
    )


# --- Roos en Louis groeien naar elkaars Frans toe (roos_06.6 §6.6.7) -----------------


def french_messages(features: pd.DataFrame) -> pd.DataFrame:
    """The French messages (by stopwords), each once, with the cleaned text as `message`.

    UnmergeMessages also brings back a quoted message a second time, under its own
    header: the same time, author and text. It is kept once.
    """
    french = features[features["is_french"] & (features["clean_message"] != "")]
    french = french.assign(message=french["clean_message"])
    return french.drop_duplicates(["ts_local", "author", "message"])


def french_units(
    french: pd.DataFrame, grouping: Grouping = GROUPING, limits: Limits = LIMITS
) -> pd.DataFrame:
    """A person's French messages less than `grouping.unit_gap_seconds` apart, joined
    into one unit.

    Units under `limits.min_unit_words` words are dropped. A unit takes its first message's time.
    `sessionize` and `merge_messages` are the teacher's (scripts/sessionize.py).
    """
    french = french.copy()
    french["session_id"] = sessionize(
        french, "ts_local", "author", grouping.unit_gap_seconds
    )
    units = merge_messages(french.sort_values("ts_local"), ["author", "session_id"])
    start = (
        french.groupby(["author", "session_id"])["ts_local"].min().rename("ts_local")
    )
    units = units.join(start, on=["author", "session_id"])
    units["n_words"] = units["message"].str.split().map(len)
    return units[units["n_words"] >= limits.min_unit_words].reset_index(drop=True)


def preprocess(text: str, patterns: Patterns = PATTERNS) -> str:
    """Lowercase, emoji out, every digit a 0, whitespace collapsed."""
    text = re.sub(patterns.emoji, " ", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"\d", "0", text)).strip()


def mask_content_words(
    text: str, word_lists: WordLists = WORD_LISTS, patterns: Patterns = PATTERNS
) -> str:
    """Every letter token that is not a function word becomes "x"; the rest stays."""
    keep = word_lists.function_words
    return re.sub(
        patterns.masked_word,
        lambda m: m.group(0) if m.group(0) in keep else word_lists.mask,
        text,
    )


def style_quarter(when: pd.Series, grouping: Grouping = GROUPING) -> pd.Series:
    """ "2024Q1"-style labels, with the thin quarters merged (`quarter_merge`)."""
    return when.dt.to_period("Q").astype(str).replace(grouping.quarter_merge)


def fragments(
    units: pd.DataFrame, grouping: Grouping = GROUPING, limits: Limits = LIMITS
) -> pd.DataFrame:
    """Every unit cut into pieces of `grouping.fragment_words` words, masked, with its
    quarter.

    A remainder under `limits.min_fragment_words` is dropped. Columns: `author`, `ts_local` (of the
    unit), `text`, `masked`, `quarter`.
    """
    size = grouping.fragment_words
    rows = []
    for author, when, text in zip(
        units["author"], units["ts_local"], units["message"], strict=True
    ):
        words = preprocess(text).split()
        for start in range(0, len(words), size):
            piece = words[start : start + size]
            if len(piece) >= limits.min_fragment_words:
                rows.append((author, when, " ".join(piece)))
    frags = pd.DataFrame(rows, columns=["author", "ts_local", "text"])
    frags["masked"] = frags["text"].map(mask_content_words)
    frags["quarter"] = style_quarter(frags["ts_local"], grouping)
    return frags


def sample_documents(
    frags: pd.DataFrame, grouping: Grouping = GROUPING, seed: int = TESTS.style.seed
) -> pd.DataFrame:
    """Per person per quarter, `docs_per_quarter` documents of `fragments_per_doc`
    random fragments.

    With replacement when the quarter has fewer than `fragments_per_doc` fragments. A
    document takes the median time of its fragments.
    """
    n_docs, per_doc = grouping.docs_per_quarter, grouping.fragments_per_doc
    rng = np.random.default_rng(seed)
    rows = []
    for (author, quarter), group in frags.groupby(["author", "quarter"], sort=True):
        replace = len(group) < per_doc
        for _ in range(n_docs):
            pick = group.iloc[rng.choice(len(group), size=per_doc, replace=replace)]
            rows.append(
                (author, quarter, pick["ts_local"].median(), " ".join(pick["masked"]))
            )
    return pd.DataFrame(rows, columns=["author", "quarter", "ts_local", "text"])


def fit_style_space(
    docs: pd.DataFrame, seed: int = TESTS.style.seed
) -> tuple[TfidfVectorizer, PCA, np.ndarray]:
    """tf-idf of character trigrams within words (char_wb), then PCA to two axes."""
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 3), min_df=5, sublinear_tf=True
    )
    weights = vectorizer.fit_transform(docs["text"]).toarray()
    pca = PCA(n_components=2, random_state=seed).fit(weights)
    return vectorizer, pca, pca.transform(weights)


def axis_loadings(
    vectorizer: TfidfVectorizer, pca: PCA, n: int = 12
) -> dict[str, dict[str, list[str]]]:
    """Per axis, the `n` trigrams with the lowest and the highest loading."""
    names = vectorizer.get_feature_names_out()
    out = {}
    for axis, component in zip(("x", "y"), pca.components_, strict=True):
        order = np.argsort(component)
        out[axis] = {
            "low": [str(names[i]) for i in order[:n]],
            "high": [str(names[i]) for i in order[::-1][:n]],
        }
    return out


def marker_score(
    vectorizer: TfidfVectorizer,
    component: np.ndarray,
    high: tuple[str, ...],
    low: tuple[str, ...],
) -> float:
    """Mean loading of the `high` markers minus that of the `low` ones.

    Missing markers are skipped. Positive: the axis already points the way its
    labels say.
    """
    index = {name: i for i, name in enumerate(vectorizer.get_feature_names_out())}

    def mean(markers: tuple[str, ...]) -> float:
        found = [component[index[m]] for m in markers if m in index]
        return float(np.mean(found)) if found else 0.0

    return mean(high) - mean(low)


def axis_signs(
    vectorizer: TfidfVectorizer, pca: PCA, word_lists: WordLists = WORD_LISTS
) -> tuple[int, int]:
    """+1 or -1 per axis, so the picture agrees with its labels (`style_markers`).

    A score of exactly 0 (no marker found) keeps the axis as PCA gave it.
    """
    x, y = (
        -1
        if marker_score(vectorizer, component, **word_lists.style_markers[axis]) < 0
        else 1
        for axis, component in zip(("x", "y"), pca.components_, strict=True)
    )
    return x, y


def half_year(when: pd.Series, grouping: Grouping = GROUPING) -> pd.Series:
    """ "2024H1"-style labels, the incomplete last half-year merged."""
    half = np.where(when.dt.month <= 6, "H1", "H2")
    return (when.dt.year.astype(str) + half).replace(grouping.half_year_merge)


def half_year_means(docs: pd.DataFrame, grouping: Grouping = GROUPING) -> pd.DataFrame:
    """Per person per half-year, the mean position of the documents, in time order."""
    frame = docs.assign(half=half_year(docs["ts_local"], grouping))
    return (
        frame.groupby(["author", "half"], sort=True)
        .agg(x=("x", "mean"), y=("y", "mean"), ts_local=("ts_local", "median"))
        .reset_index()
    )


def permutation_spearman(
    order: np.ndarray, values: np.ndarray, test: ShuffleTest = TESTS.style
) -> tuple[float, float]:
    """Spearman's rho between `order` and `values`, and a two-sided shuffle p.

    p = (shuffles of `order` with |rho| at least as large + 1) / (n + 1).
    """
    n = test.n_shuffle
    rng = np.random.default_rng(test.seed)
    x = stats.rankdata(order)
    y = stats.rankdata(values)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    rho = float(np.mean(x * y))
    shuffled = np.array([rng.permutation(x) for _ in range(n)])
    null = (shuffled * y).mean(axis=1)
    return rho, float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n + 1))


def monthly_trend(
    frags: pd.DataFrame,
    positions: np.ndarray,
    author: str,
    axis: int,
    test: ShuffleTest = TESTS.style,
    limits: Limits = LIMITS,
) -> Trend:
    """Does one person's position on `axis` move with the months?

    Per month the mean position of their fragments (months with
    `limits.min_month_fragments` or more), and the shuffle Spearman between month and
    position.
    """
    mine = frags["author"].to_numpy() == author
    month = frags.loc[mine, "ts_local"].dt.to_period("M")
    per_month = pd.DataFrame(
        {"month": month.to_numpy(), "pos": positions[mine, axis]}
    ).groupby("month")["pos"]
    means = per_month.mean()[per_month.size() >= limits.min_month_fragments]
    rho, p = permutation_spearman(np.arange(len(means)), means.to_numpy(), test)
    return Trend(rho=rho, p=p, n_months=len(means))


def quarterly_similarity(
    docs: pd.DataFrame, vectorizer: TfidfVectorizer, categories: Categories = CATEGORIES
) -> dict[str, float]:
    """Per quarter, the cosine between Roos's and Louis's mean tf-idf profile."""
    weights = vectorizer.transform(docs["text"]).toarray()
    out = {}
    for quarter in sorted(docs["quarter"].unique()):
        in_quarter = docs["quarter"] == quarter
        roos, louis = (
            weights[(in_quarter & (docs["author"] == a)).to_numpy()].mean(axis=0)
            for a in categories.authors
        )
        out[str(quarter)] = float(
            roos @ louis / (np.linalg.norm(roos) * np.linalg.norm(louis))
        )
    return out


def style_map(
    features: pd.DataFrame,
    test: ShuffleTest = TESTS.style,
    grouping: Grouping = GROUPING,
    limits: Limits = LIMITS,
) -> tuple[pd.DataFrame, pd.DataFrame, StyleStats]:
    """Everything the les-6 figure draws: documents, half-year means, and the numbers.

    French messages -> units -> masked fragments -> 12 documents per person per
    quarter -> tf-idf on char_wb trigrams -> PCA(2), axes turned to agree with their
    labels. Tests: every fragment projected onto the axes, Spearman between month and
    position (Roos on y, Louis on x), and the trend in the quarterly cosine between
    the two profiles, all with shuffle p-values.
    """
    french = french_messages(features)
    units = french_units(french, grouping, limits)
    frags = fragments(units, grouping, limits)
    docs = sample_documents(frags, grouping, test.seed)
    vectorizer, pca, coords = fit_style_space(docs, test.seed)
    signs = axis_signs(vectorizer, pca)
    docs[["x", "y"]] = coords * np.asarray(signs)
    frag_coords = pca.transform(
        vectorizer.transform(frags["masked"]).toarray()
    ) * np.asarray(signs)
    similarity = quarterly_similarity(docs, vectorizer)
    sim_rho, sim_p = permutation_spearman(
        np.arange(len(similarity)), np.array(list(similarity.values())), test
    )
    result = StyleStats(
        first_day=str(french["ts_local"].min().date()),
        last_day=str(french["ts_local"].max().date()),
        n_units={str(a): int(n) for a, n in units["author"].value_counts().items()},
        n_fragments={str(a): int(n) for a, n in frags["author"].value_counts().items()},
        n_docs=len(docs),
        n_trigrams=len(vectorizer.vocabulary_),
        explained=[float(v) for v in pca.explained_variance_ratio_],
        signs=list(signs),
        loadings=axis_loadings(vectorizer, pca),
        roos_trend=monthly_trend(frags, frag_coords, "Roos", 1, test, limits),
        louis_trend=monthly_trend(frags, frag_coords, "Louis", 0, test, limits),
        similarity=similarity,
        similarity_rho=sim_rho,
        similarity_p=sim_p,
        n_shuffle=test.n_shuffle,
    )
    columns = ["author", "quarter", "ts_local", "x", "y"]
    return docs[columns], half_year_means(docs, grouping), result


def run(paths: Paths, tests: Tests = TESTS) -> None:
    """Read both stage-2 files, write the eleven small stage-3 files."""
    features = pd.read_parquet(paths.features)

    language_share(features).to_parquet(paths.language_share)
    topics = weekly_topics(features)
    topics.to_parquet(paths.weekly_topics)
    b = topic_balance_stats(topics, "climbing", tests.topic_balance)
    paths.topic_stats.write_text(b.model_dump_json(indent=2))
    logger.info(
        f"aggregate: child minus climbing, {b.mean_gap_before:+.3f} -> "
        f"{b.mean_gap_after:+.3f}, shuffle p <= {b.p_value_greater:.4f}, "
        f"{b.n_runs_reaching_after} of {b.n_runs} runs reach it"
    )

    weekly = weekly_symptoms(features)
    weekly.to_parquet(paths.weekly_symptoms)

    louis = louis_messages(features)
    french = weekly_french(louis)
    french.to_parquet(paths.weekly_french)
    length = french_length_stats(louis, french)
    paths.french_stats.write_text(length.model_dump_json(indent=2))
    logger.info(
        f"aggregate: French shorter in {length.n_weeks_below} of "
        f"{length.n_weeks_compared} long-distance weeks, median gap "
        f"{length.median_difference:+.1f} words, Wilcoxon p = "
        f"{length.wilcoxon_p_less:.1e}"
    )

    unmerged = pd.read_parquet(paths.unmerged_features)
    theme_shares(unmerged).to_parquet(paths.home_shares)
    home = home_stats(unmerged, tests.home)
    paths.home_stats.write_text(home.model_dump_json(indent=2))
    logger.info(
        f"aggregate: thuis-score {home.place} minus the others "
        f"{home.messages.gap_observed:+.0%}, shuffle p <= "
        f"{home.messages.p_value_greater:.4f} (messages), "
        f"{home.days.p_value_greater:.4f} (days)"
    )

    docs, means, style = style_map(unmerged, tests.style)
    docs.to_parquet(paths.style_docs)
    means.to_parquet(paths.style_means)
    paths.style_stats.write_text(style.model_dump_json(indent=2))
    logger.info(
        f"aggregate: style map, Roos rho {style.roos_trend.rho:+.2f}, Louis rho "
        f"{style.louis_trend.rho:+.2f}, similarity rho "
        f"{style.similarity_rho:+.2f} (p = {style.similarity_p:.4f})"
    )

    result = symptom_stats(weekly, tests.symptoms)
    paths.symptom_stats.write_text(result.model_dump_json(indent=2))
    logger.success(
        f"aggregate: hard weeks {result.mean_hard:.1f} vs normal "
        f"{result.mean_normal:.1f} symptom messages a week, shuffle p <= "
        f"{result.p_value_greater:.4f} -> {paths.processed}"
    )

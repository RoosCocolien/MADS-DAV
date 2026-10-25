"""Stage 3 — aggregate: change the grain from messages to periods and weeks, and fit.

    2_features.parquet
      -> language_share        % per language per move period  -> 3_language_share.parquet
      -> weekly_topics         topic share per calendar week    -> 3_weekly_topics.parquet
           -> topic_balance_stats   child minus sport, before vs after the test:
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

Everything after this stage works on a few hundred rows instead of 63,000 messages,
and the slowest step of the whole pipeline (10,000 shuffles) runs here, once. The
figures stage only reads these seven small files, so a title can change without
re-running any of it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from goad_toolkit.analytics import NullDistribution
from goad_toolkit.datatransforms import Filter, Pipeline
from loguru import logger
from scipy import optimize, stats

from roos_pipeline import settings
from roos_pipeline.features import PregnancyWeek
from roos_pipeline.settings import Paths

if TYPE_CHECKING:
    # scipy has no public name for a frozen distribution; only needed for the type.
    from scipy.stats._distn_infrastructure import rv_discrete_frozen

TOPICS = ["child", "climbing", "running", "sports"]
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


def weekly_topics(features: pd.DataFrame) -> pd.DataFrame:
    """Per calendar week: each topic's share of that week's messages, and the counts.

    A share, not a count (03.3.2's rule): message volume itself halves after the move,
    and a count would re-tell that drop instead of a change in topic. A week without
    messages keeps a NaN share, so the chart breaks the line instead of inventing a
    zero.
    """
    by_week = features.set_index("ts_local")[TOPICS].resample("W")
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
    n_shuffle: int = settings.N_SHUFFLE,
    run_weeks: int = settings.RUN_WEEKS,
) -> dict:
    """Did the balance between child and sport tip at the positive test, beyond chance?

    One number per week: child's share minus `sport`'s share of that week's messages
    (Roos's choice, 2026-10-06: one test on the gap, "the balance tipped", rather than
    two separate tests). Weeks before the test against weeks from the test on; the one
    week that contains the test is left out, because it is half of each.

    Two checks, the same pair as roos_04.6d:

    1. **Shuffle** the before/after label over all weeks (`NullDistribution`): how
       often does a random split produce a shift this large? This treats weeks as
       interchangeable, which neighbouring weeks are not quite.
    2. **Keep the order of time:** every run of `run_weeks` consecutive weeks before
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
                [week_end < settings.TEST_DATE, week_start >= settings.TEST_DATE],
                [BEFORE_TEST, AFTER_TEST],
                default="",
            ),
        }
    )
    frame = frame[(frame["period"] != "") & frame["gap"].notna()]
    before = frame.loc[frame["period"] == BEFORE_TEST, "gap"]
    after = frame.loc[frame["period"] == AFTER_TEST, "gap"]

    null = NullDistribution(mean_shift, n_iter=n_shuffle, seed=settings.SEED).run(
        frame, label="period"
    )
    # Rolling over the before-weeks in calendar order; with fewer weeks than one run
    # there is no run at all, and the result says so instead of inventing one.
    runs = before.reset_index(drop=True).rolling(run_weeks).mean().dropna()
    return {
        "sport": sport,
        "n_before": len(before),
        "n_after": len(after),
        "mean_gap_before": float(before.mean()),
        "mean_gap_after": float(after.mean()),
        "shift_observed": float(null.observed),
        "n_shuffle": n_shuffle,
        "n_shuffles_reaching_shift": int((null.null >= null.observed).sum()),
        "p_value_greater": float(null.p_value("greater")),
        "run_weeks": run_weeks,
        "n_runs": len(runs),
        "max_run_gap": float(runs.max()) if len(runs) else None,
        "n_runs_reaching_after": int((runs >= after.mean()).sum()),
    }


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


def weekly_symptoms(features: pd.DataFrame) -> pd.DataFrame:
    """Symptom messages per pregnancy week, every week a full 7 days.

    Weeks are pregnancy weeks (see `PregnancyWeek`), so the hard period is exactly
    weeks 5-14. The partial first and last block are dropped: a 3-day "week" would
    look like a quiet week. `group` labels the two sides of the comparison.
    """
    daily = Pipeline().add(PregnancyWeek, column="day", start=settings.PREGNANCY_START)
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
            weekly.index <= settings.LAST_NORMAL_WEEK,
            weekly.index.isin(settings.HARD_WEEKS),
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

    def nll(theta):  # log-parameters keep mu and r positive
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


def symptom_stats(weekly: pd.DataFrame, n_shuffle: int = settings.N_SHUFFLE) -> dict:
    """Fit both groups, and shuffle the group label to see what chance produces.

    Returns plain numbers only (no scipy objects), so the result can be written to
    JSON and the figure stage can rebuild each curve from `mu` and `r`.
    """
    two_groups = weekly[weekly["group"].isin([NORMAL, HARD])]
    normal = two_groups.loc[two_groups["group"] == NORMAL, "either"].to_numpy(float)
    hard = two_groups.loc[two_groups["group"] == HARD, "either"].to_numpy(float)
    if len(hard) != len(settings.HARD_WEEKS):
        raise ValueError(
            f"Expected {len(settings.HARD_WEEKS)} full hard weeks, got {len(hard)}"
        )

    normal_fit, hard_fit = fit_daily(normal), fit_daily(hard)
    null = NullDistribution(mean_gap, n_iter=n_shuffle, seed=settings.SEED).run(
        two_groups[["either", "group"]], label="group"
    )
    return {
        "n_normal": len(normal),
        "n_hard": len(hard),
        "mean_normal": float(normal.mean()),
        "mean_hard": float(hard.mean()),
        "max_count": int(max(normal.max(), hard.max())),
        "normal_mu": float(normal_fit["nb_mu"]),
        "normal_r": float(normal_fit["nb_r"]),
        "hard_mu": float(hard_fit["nb_mu"]),
        "hard_r": float(hard_fit["nb_r"]),
        "hard_delta_aic_vs_poisson": float(
            hard_fit["poisson_aic"] - hard_fit["nb_aic"]
        ),
        "n_weeks_shuffled": len(two_groups),
        "n_shuffle": n_shuffle,
        "gap_observed": float(null.observed),
        "n_shuffles_reaching_gap": int((null.null >= null.observed).sum()),
        "p_value_greater": float(null.p_value("greater")),
    }


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
    louis: pd.DataFrame, min_messages: int = settings.MIN_MESSAGES
) -> pd.DataFrame:
    """Per calendar week while we lived apart: Louis's French share and message length.

    The French share is over his classified messages only (`is_french` is NaN for
    "unknown"); words per message count all of them, the short ones included. Only
    weeks that end before the move (the week of the move is half of each), and with
    at least `min_messages` messages: one long message makes a week of 3 look long.
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
    keep = (weekly["week"] < settings.MOVE_DATE) & (
        weekly["n_messages"] >= min_messages
    )
    return weekly[keep].dropna(subset=["french_share"]).reset_index(drop=True)


def within_week_differences(
    messages: pd.DataFrame, min_per_group: int = settings.MIN_PER_GROUP
) -> pd.Series:
    """Per week: median words of his French messages minus median of his other ones.

    A week needs `min_per_group` classified messages in each group, otherwise one
    message would be "the median". Below 0: his French messages were shorter that week.
    """
    classified = messages[messages["is_french"].notna()]
    group = np.where(classified["is_french"] == 1.0, "French", "other")
    per_week = (
        classified.groupby(["week", group])["n_words"].agg(["median", "size"]).unstack()
    )
    both = per_week[
        (per_week[("size", "French")] >= min_per_group)
        & (per_week[("size", "other")] >= min_per_group)
    ]
    return (both[("median", "French")] - both[("median", "other")]).rename("difference")


def french_length_stats(louis: pd.DataFrame, weekly: pd.DataFrame) -> dict:
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
    diff = within_week_differences(apart)
    wilcoxon = stats.wilcoxon(diff, alternative="less")
    b, a = np.polyfit(np.log(weekly["french_share"]), weekly["median_words"], 1)
    counts = apart["language"].value_counts()
    return {
        "n_weeks": len(weekly),
        "first_day": str((weekly["week"].min() - pd.Timedelta(days=6)).date()),
        "last_week": str(weekly["week"].max().date()),
        "n_messages": len(apart),
        "language_counts": {
            lang: int(counts.get(lang, 0)) for lang in [*settings.LANGUAGES, "unknown"]
        },
        "fit_a": float(a),
        "fit_b": float(b),
        "words_per_doubling": float(b * np.log(2)),
        "n_weeks_compared": len(diff),
        "n_weeks_below": int((diff < 0).sum()),
        "median_difference": float(diff.median()),
        "wilcoxon_p_less": float(wilcoxon.pvalue),
    }


def run(paths: Paths, n_shuffle: int = settings.N_SHUFFLE) -> None:
    """Read 2_features.parquet, write the seven small stage-3 files."""
    features = pd.read_parquet(paths.features)

    language_share(features).to_parquet(paths.language_share)
    topics = weekly_topics(features)
    topics.to_parquet(paths.weekly_topics)
    balance = {
        sport: topic_balance_stats(topics, sport, n_shuffle=n_shuffle)
        for sport in ["climbing", "sports"]
    }
    paths.topic_stats.write_text(json.dumps(balance, indent=2))
    for sport, b in balance.items():
        logger.info(
            f"aggregate: child minus {sport}, {b['mean_gap_before']:+.3f} -> "
            f"{b['mean_gap_after']:+.3f}, shuffle p <= {b['p_value_greater']:.4f}, "
            f"{b['n_runs_reaching_after']} of {b['n_runs']} runs reach it"
        )

    weekly = weekly_symptoms(features)
    weekly.to_parquet(paths.weekly_symptoms)

    louis = louis_messages(features)
    french = weekly_french(louis)
    french.to_parquet(paths.weekly_french)
    length = french_length_stats(louis, french)
    paths.french_stats.write_text(json.dumps(length, indent=2))
    logger.info(
        f"aggregate: French shorter in {length['n_weeks_below']} of "
        f"{length['n_weeks_compared']} long-distance weeks, median gap "
        f"{length['median_difference']:+.1f} words, Wilcoxon p = "
        f"{length['wilcoxon_p_less']:.1e}"
    )

    result = symptom_stats(weekly, n_shuffle=n_shuffle)
    paths.symptom_stats.write_text(json.dumps(result, indent=2))
    logger.success(
        f"aggregate: hard weeks {result['mean_hard']:.1f} vs normal "
        f"{result['mean_normal']:.1f} symptom messages a week, shuffle p <= "
        f"{result['p_value_greater']:.4f} -> {paths.processed}"
    )

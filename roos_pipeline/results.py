"""The stage-3 JSON files, as typed models: what aggregate writes and figures reads.

Before 2026-11-13 these were plain `dict`s: `stats: dict` in nine figure functions said
nothing about which numbers were in it, and a typo such as `stats["mean_hrad"]` only
showed up as a KeyError at run time. Now ty knows every field, and reading a file checks
it (`SymptomStats.model_validate_json(...)`): a stale file from an older run fails at the
door instead of halfway through a figure.

Classes without behaviour: fields only, like the settings.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Result(BaseModel):
    """The base of every stage-3 result: read-only once made."""

    model_config = ConfigDict(frozen=True)


class TopicBalanceStats(Result):
    """Les 3: child minus climbing, before vs. after the test -> 3_topic_stats.json."""

    sport: str
    n_before: int
    n_after: int
    mean_gap_before: float
    mean_gap_after: float
    shift_observed: float
    n_shuffle: int
    n_shuffles_reaching_shift: int
    p_value_greater: float
    run_weeks: int
    n_runs: int
    max_run_gap: float | None  # None: too few weeks for a single run
    n_runs_reaching_after: int


class SymptomStats(Result):
    """Les 4: two fitted negative binomials and the shuffle -> 3_symptom_stats.json."""

    n_normal: int
    n_hard: int
    mean_normal: float
    mean_hard: float
    max_count: int
    normal_mu: float
    normal_r: float
    hard_mu: float
    hard_r: float
    hard_delta_aic_vs_poisson: float
    n_weeks_shuffled: int
    n_shuffle: int
    gap_observed: float
    n_shuffles_reaching_gap: int
    p_value_greater: float


class FrenchStats(Result):
    """Les 5: the log fit and the within-week test -> 3_french_stats.json."""

    n_weeks: int
    first_day: str  # ISO date
    last_week: str  # ISO date
    n_messages: int
    language_counts: dict[str, int]  # per language code, plus "unknown"
    fit_a: float
    fit_b: float
    words_per_doubling: float
    n_weeks_compared: int
    n_weeks_below: int
    median_difference: float
    wilcoxon_p_less: float


class Trend(Result):
    """One person's style moving with the months (les 6)."""

    rho: float
    p: float
    n_months: int


class StyleStats(Result):
    """Les 6: the style map's numbers -> 3_style_stats.json."""

    first_day: str
    last_day: str
    n_units: dict[str, int]
    n_fragments: dict[str, int]
    n_docs: int
    n_trigrams: int
    explained: list[float]
    signs: list[int]
    loadings: dict[str, dict[str, list[str]]]
    roos_trend: Trend
    louis_trend: Trend
    similarity: dict[str, float]  # per quarter, in time order
    similarity_rho: float
    similarity_p: float
    n_shuffle: int


class HomeTestResult(Result):
    """Les 7's shuffle test with one unit (a message or a day)."""

    n_place: int
    n_against: int
    gap_observed: float
    n_shuffles_reaching_gap: int
    p_value_greater: float


class HomeStats(Result):
    """Les 7: the place against the others, per message and per day -> 3_home_stats.json."""

    place: str
    against: list[str]
    n_shuffle: int
    messages: HomeTestResult
    days: HomeTestResult

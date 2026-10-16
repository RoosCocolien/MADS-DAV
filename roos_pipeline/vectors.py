"""Lesson 6 — does Louis say different things in French than in English?

Used by `notebooks/lesson6/roos_06.5-louis-fr-vs-en.ipynb`. The steps, in order:

    2_features.parquet (one row per message)
      -> StripQuotedReply      the quoted "[18-12-2025, 09:42:27] Roos: ..." tail removed
      -> WeekStart             the Monday each message's week starts on
      -> language_units()      grain changes: one row per (session, language), 6.4's unit
      -> WordCount             n_words and a length band per unit
    -> embed, k-means (in the notebook) -> cluster_contrast() -> the two plot classes

Every step class is a goad `TransformBase`, like the ones in `features.py`, so it slots
into `Pipeline.add(...)`. `language_units` and `cluster_contrast` change the grain, so
they are functions, the same split `aggregate.py` makes.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from goad_toolkit.datatransforms import TransformBase
from goad_toolkit.visualizer import BasePlot

from roos_pipeline import settings
from scripts.sessionize import merge_messages, sessionize


class StripQuotedReply(TransformBase):
    """Keep only what the sender typed: cut a reply's quoted message off the end.

    A reply that quotes an earlier message carries that message as
    "[28-08-2024, 18:21:37] Roos: Groetjes", so without this step Roos's words would be
    embedded as Louis's (found in roos_05.6). The invisible left-to-right mark (U+200E)
    WhatsApp puts in front of the quote goes too.
    """

    def transform(
        self,
        data: pd.DataFrame,
        column: str = "message",
        pattern: str = settings.QUOTED_REPLY_PATTERN,
    ) -> pd.DataFrame:
        data[column] = (
            data[column]
            .fillna("")
            .str.split(pattern, n=1, regex=True)
            .str[0]
            .str.replace("‎", "", regex=False)
            .str.strip()
        )
        return data


class WeekStart(TransformBase):
    """The Monday 00:00 of the week a message falls in, as a timestamp."""

    def transform(
        self, data: pd.DataFrame, column: str = "ts_local", feature: str = "week"
    ) -> pd.DataFrame:
        data[feature] = data[column].dt.to_period("W-SUN").dt.start_time
        return data


class WordCount(TransformBase):
    """Words per text, and which length band that falls in.

    The band is what the shuffle test holds fixed: French units are much shorter than
    English ones, and a short text lands in a different part of the embedding space
    whatever it says.
    """

    def transform(
        self,
        data: pd.DataFrame,
        column: str = "message",
        bands: tuple[int, ...] = settings.LENGTH_BANDS,
    ) -> pd.DataFrame:
        data["n_words"] = data[column].fillna("").str.count(r"\S+")
        edges = [0, *bands, np.inf]
        labels = [f"{lo + 1}-{hi}" for lo, hi in zip(edges[:-2], edges[1:-1])]
        labels.append(f"{bands[-1] + 1}+")
        data["length_band"] = pd.cut(data["n_words"], edges, labels=labels)
        return data


def language_units(
    messages: pd.DataFrame,
    threshold: float,
    languages: list[str] = settings.COMPARE_LANGUAGES,
) -> pd.DataFrame:
    """One row per (session, language): 6.4's unit, split by language.

    Sessions come first, from *all* of one author's messages, so a switch to French in the
    middle of a conversation does not start a new session. Then each session's French
    messages are joined into one unit and its English messages into another. A unit takes
    the week of its first message.

    Args:
        messages: one author's messages, with `ts_local`, `author`, `message`, `language`
            and `week`.
        threshold: the session gap in seconds, from `fit_session_threshold`.
        languages: which languages to keep; the others (Dutch, "unknown") still count
            for where a session starts and ends.

    Returns:
        `session_id`, `language`, `message` (joined), `n_messages`, `week`.
    """
    if messages["author"].nunique() != 1:
        raise ValueError("language_units expects one author's messages")
    ordered = messages.sort_values("ts_local").copy()
    ordered["session_id"] = sessionize(ordered, "ts_local", "author", threshold)
    kept = ordered[ordered["language"].isin(languages)]
    units = merge_messages(kept, ["session_id", "language"])
    first_week = kept.groupby(["session_id", "language"])["week"].min()
    return units.join(first_week, on=["session_id", "language"])


def cluster_shares(
    units: pd.DataFrame, cluster: str = "cluster", group: str = "language"
) -> pd.DataFrame:
    """Share of each group's units that fall in each cluster (columns sum to 1)."""
    return pd.crosstab(units[cluster], units[group], normalize="columns")


def cluster_contrast(
    units: pd.DataFrame,
    cluster: str = "cluster",
    group: str = "language",
    strata: tuple[str, ...] = ("week", "length_band"),
    pair: tuple[str, str] = ("fr", "en"),
    n_shuffle: int = settings.N_SHUFFLE,
    seed: int = settings.SEED,
) -> pd.DataFrame:
    """French share − English share per cluster, against a shuffle that keeps the strata.

    The shuffle swaps the language labels *within* each week × length band. A week keeps
    its number of French and English units, a band keeps its own, so the shuffled
    difference is what week and length alone would produce. The real difference minus
    the shuffled mean (`beyond_strata`) is the part left over for the language itself.

    Returns:
        One row per cluster: both shares, the raw difference, the shuffle's mean,
        `beyond_strata`, and a two-sided p-value (with the +1 correction, as in 04.6).
    """
    first, second = pair
    data = units[units[group].isin(pair)].reset_index(drop=True)
    clusters = np.sort(data[cluster].unique())
    onehot = (data[cluster].to_numpy()[:, None] == clusters[None, :]).astype(float)
    labels = data[group].to_numpy()

    def difference(lab: np.ndarray) -> np.ndarray:
        return onehot[lab == first].mean(axis=0) - onehot[lab == second].mean(axis=0)

    observed = difference(labels)
    rng = np.random.default_rng(seed)
    # A shuffle within every block at once: sorting by (block, random key) lists each
    # block's rows in a random order, sorting by block alone in a fixed one. Both lists
    # hold the same rows per block, so pairing them up permutes within blocks only.
    block = data.groupby(list(strata), observed=True).ngroup().to_numpy()
    fixed = np.argsort(block, kind="stable")
    null = np.empty((n_shuffle, len(clusters)))
    shuffled = labels.copy()
    for i in range(n_shuffle):
        shuffled[fixed] = labels[np.lexsort((rng.random(len(block)), block))]
        null[i] = difference(shuffled)

    centre = null.mean(axis=0)
    extreme = (np.abs(null - centre) >= np.abs(observed - centre)).sum(axis=0)
    shares = cluster_shares(data, cluster, group)
    return pd.DataFrame(
        {
            f"share_{first}": shares[first].reindex(clusters).to_numpy(),
            f"share_{second}": shares[second].reindex(clusters).to_numpy(),
            "difference": observed,
            "shuffle_mean": centre,
            "beyond_strata": observed - centre,
            "p": (extreme + 1) / (n_shuffle + 1),
        },
        index=pd.Index(clusters, name=cluster),
    )


def weekly_cluster_shares(
    units: pd.DataFrame,
    cluster: str = "cluster",
    group: str = "language",
    min_units: int = settings.MIN_UNITS_PER_WEEK,
) -> pd.DataFrame:
    """Per week and language, the share of units in each cluster.

    A week only counts for a language if it has at least `min_units` units in it: one
    French unit in a week would make that week's French share 0% or 100%.

    Returns:
        Long format: `week`, `language`, `cluster`, `share`, `n_units`.
    """
    # crosstab, not groupby: a cluster with no units that week is a 0, not missing.
    counts = pd.crosstab([units["week"], units[group]], units[cluster])
    n_units = counts.sum(axis=1)
    kept = n_units >= min_units
    shares = counts[kept].div(n_units[kept], axis=0)
    long = shares.reset_index().melt(
        id_vars=["week", group], var_name=cluster, value_name="share"
    )
    return long.join(n_units.rename("n_units"), on=["week", group])


def cluster_examples(
    texts: pd.Series,
    vectors: np.ndarray,
    labels: np.ndarray,
    centres: np.ndarray,
    n: int = 8,
    width: int = 120,
) -> dict[int, list[str]]:
    """The `n` texts closest to each cluster's centre: what a cluster is "about".

    Closest by cosine (the vectors and centres are normalised first), so these are the
    most typical members, not a random draw.
    """
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    centre = centres / np.linalg.norm(centres, axis=1, keepdims=True)
    examples = {}
    for k in range(len(centres)):
        members = np.flatnonzero(labels == k)
        closest = members[np.argsort(-(unit[members] @ centre[k]))[:n]]
        examples[k] = [texts.iloc[i][:width] for i in closest]
    return examples


# --- plots -------------------------------------------------------------------------


class LanguageDumbbellPlot(BasePlot):
    """Two dots per cluster, French and English, joined by a line: how far apart they are.

    Built for two groups × a handful of categories, where the *gap* is the finding: a
    grouped bar chart makes the eye compare bar heights across pairs, a dumbbell puts each
    pair's gap on one line. Clusters run top to bottom in the order given; the p-value of
    `cluster_contrast` is written at the right.
    """

    def build(
        self,
        contrast: pd.DataFrame,
        names: dict[int, str],
        pair: tuple[str, str] = ("fr", "en"),
        colors: dict[str, str] | None = None,
        pair_labels: dict[str, str] | None = None,
    ):
        if self.ax is None:
            raise ValueError("create_figure() must run before build()")
        colors = colors or {
            pair[0]: settings.FRENCH_COLOR,
            pair[1]: settings.ENGLISH_COLOR,
        }
        pair_labels = pair_labels or {lang: lang for lang in pair}
        rows = list(contrast.index)
        y = np.arange(len(rows))[::-1]
        first, second = (contrast[f"share_{lang}"].to_numpy() * 100 for lang in pair)
        self.ax.hlines(
            y, first, second, color=settings.LIGHT_GREY, linewidth=3, zorder=1
        )
        for lang, values in zip(pair, (first, second)):
            self.ax.scatter(
                values, y, color=colors[lang], s=70, zorder=2, label=pair_labels[lang]
            )
        right = max(first.max(), second.max()) * 1.08
        for yi, p in zip(y, contrast["p"]):
            self.ax.text(
                right, yi, f"p = {p:.3f}", va="center", color="dimgray", fontsize=9
            )
        self.ax.set_yticks(y, [names.get(k, f"cluster {k}") for k in rows])
        self.ax.set_xlim(0, right * 1.15)
        self.ax.spines[["top", "right"]].set_visible(False)
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
        return self.fig, self.ax


class WeeklyLanguagePanels(BasePlot):
    """One panel per cluster: that cluster's weekly share among French and among English
    units, as two lines over the same weeks.

    Thin dots are the real weeks, the line a rolling average; a week missing for one
    language (fewer than `min_units` units) is a gap in that line, not a zero. Call with
    `n_plots=` the number of clusters, panel titles in `PlotSettings.subplot_titles`.
    """

    def build(
        self,
        weekly: pd.DataFrame,
        clusters: list[int],
        pair: tuple[str, str] = ("fr", "en"),
        colors: dict[str, str] | None = None,
        pair_labels: dict[str, str] | None = None,
        smooth_weeks: int = settings.SMOOTH_WEEKS,
    ):
        if self.fig is None:
            raise ValueError("create_figure() must run before build()")
        axes = [ax for ax in self.fig.axes if ax.get_visible()]
        if len(axes) != len(clusters):
            raise ValueError(f"{len(clusters)} clusters need n_plots={len(clusters)}")
        colors = colors or {
            pair[0]: settings.FRENCH_COLOR,
            pair[1]: settings.ENGLISH_COLOR,
        }
        pair_labels = pair_labels or {lang: lang for lang in pair}
        weeks = pd.Index(np.sort(weekly["week"].unique()))
        for ax, k in zip(axes, clusters):
            for lang in pair:
                series = (
                    weekly[(weekly["cluster"] == k) & (weekly["language"] == lang)]
                    .set_index("week")["share"]
                    .reindex(weeks)
                    * 100
                )
                trend = series.rolling(smooth_weeks, min_periods=2, center=True).mean()
                ax.scatter(weeks, series, color=colors[lang], s=6, alpha=0.35)
                ax.plot(
                    weeks,
                    trend,
                    color=colors[lang],
                    linewidth=2,
                    label=pair_labels[lang],
                )
            ax.set_ylim(0, None)
            ax.spines[["top", "right"]].set_visible(False)
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axes[0].legend(loc="upper left", frameon=False)
        return self.fig, axes

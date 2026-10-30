"""Lesson 6 — does Louis say different things in French than in English?

Used by `notebooks/lesson6/roos_06.5-louis-fr-vs-en.ipynb`. The steps, in order:

    2_features.parquet (one row per message)
      -> StripQuotedReply      the quoted "[18-12-2025, 09:42:27] Roos: ..." tail removed
      -> WeekStart             the Monday each message's week starts on
      -> language_units()      grain changes: one row per (session, language), 6.4's unit
      -> WordCount             n_words and a length band per unit
    -> embed, k-means (in the notebook) -> cluster_keywords() for a first name per cluster
    -> cluster_contrast() (shuffle test), contrast_intervals() (weeks resampled) -> the plot classes
    -> topic_labels()          6.5.10: a topic per unit from word lists, apart from the encoder;
                               neighbour_agreement() checks it against the vectors

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
from sklearn.feature_extraction.text import CountVectorizer

from roos_pipeline import settings
from roos_pipeline.features import normalise
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


def cluster_keywords(
    texts: pd.Series,
    labels: np.ndarray,
    n: int = 6,
    min_units: int = 5,
    max_share: float = 0.05,
) -> dict[int, list[str]]:
    """The words that set each cluster apart from the others.

    Per word: the share of the cluster's units that use it (`inside`) against the share of
    the other units (`outside`). The score, inside × log(inside / outside), is high for a
    word that is common in the cluster *and* much more common there than elsewhere.

    Words in more than `max_share` of all units are dropped first. Without that, "the",
    "you" and "and" topped every English-heavy cluster: they are frequent everywhere, but
    a cluster with more English simply has more of them. Words of one or two letters and
    numbers never count.

    A first draft of a name, not the name: read the examples (`cluster_examples`) too.

    Args:
        texts: one text per unit, in the same order as `labels`.
        labels: the cluster of each unit.
        n: words per cluster.
        min_units: a word must appear in this many units; rarer ones are mostly typos and
            names, which would otherwise top a small cluster.
        max_share: words in a larger share of all units are too common to tell clusters apart.
    """
    vectorizer = CountVectorizer(
        token_pattern=settings.KEYWORD_PATTERN,
        binary=True,
        min_df=min_units,
        max_df=max_share,
    )
    used = vectorizer.fit_transform(texts.fillna("").str.lower())
    words = vectorizer.get_feature_names_out()
    keywords = {}
    for k in np.sort(np.unique(labels)):
        inside = np.asarray(used[labels == k].mean(axis=0)).ravel()
        outside = np.asarray(used[labels != k].mean(axis=0)).ravel()
        score = inside * np.log((inside + 1e-3) / (outside + 1e-3))
        keywords[int(k)] = [str(words[i]) for i in np.argsort(-score)[:n]]
    return keywords


def contrast_intervals(
    units: pd.DataFrame,
    cluster: str = "cluster",
    group: str = "language",
    pair: tuple[str, str] = ("fr", "en"),
    n_boot: int = 2000,
    seed: int = settings.SEED,
) -> pd.DataFrame:
    """French share − English share per cluster, with how much it moves between weeks.

    Two checks on one number, both in percentage points:

    - a 95% interval from resampling *weeks* (with replacement): the units of one week
      are not independent (one busy week of planning fills one cluster), so the week,
      not the unit, is what gets drawn;
    - the difference in odd and in even months separately: two halves of the data that
      each run over the whole period. If a cluster's difference is real, both halves show it.

    Returns:
        One row per cluster: `difference`, `low`, `high`, `odd_months`, `even_months`.
    """
    first, second = pair
    data = units[units[group].isin(pair)].reset_index(drop=True)
    clusters = np.sort(data[cluster].unique())

    def difference(rows: pd.DataFrame) -> np.ndarray:
        shares = pd.crosstab(rows[cluster], rows[group], normalize="columns")
        shares = shares.reindex(index=clusters, columns=list(pair), fill_value=0)
        return (shares[first] - shares[second]).to_numpy() * 100

    # Per week, units per cluster for each language; a resample is then a weighted sum.
    weeks = np.sort(data["week"].unique())
    counts = {
        lang: pd.crosstab(data.loc[data[group] == lang, "week"], data[cluster])
        .reindex(index=weeks, columns=clusters, fill_value=0)
        .to_numpy()
        for lang in pair
    }
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(clusters)))
    for i in range(n_boot):
        weight = np.bincount(
            rng.integers(len(weeks), size=len(weeks)), minlength=len(weeks)
        )
        a, b = (weight @ counts[lang] for lang in pair)
        boot[i] = (a / max(a.sum(), 1) - b / max(b.sum(), 1)) * 100

    odd = data["week"].dt.month % 2 == 1
    return pd.DataFrame(
        {
            "difference": difference(data),
            "low": np.percentile(boot, 2.5, axis=0),
            "high": np.percentile(boot, 97.5, axis=0),
            "odd_months": difference(data[odd]),
            "even_months": difference(data[~odd]),
        },
        index=pd.Index(clusters, name=cluster),
    )


def topic_labels(
    texts: pd.Series,
    terms: dict[str, tuple[str, ...]] = settings.UNIT_TOPIC_TERMS,
    both: str = "both",
    none: str = "other",
) -> pd.Series:
    """One topic per unit from word lists, without the encoder: the colour route of 6.5.10.

    A unit with a term from exactly one list gets that list's name, with terms from two
    lists `both`, with none `none`. Matched on `normalise`d text, so the terms are written
    without accents.
    """
    normalised = texts.fillna("").map(normalise)
    hits = pd.DataFrame(
        {
            name: normalised.str.contains("|".join(f"(?:{t})" for t in patterns))
            for name, patterns in terms.items()
        }
    )
    labels = pd.Series(none, index=texts.index)
    n_hits = hits.sum(axis=1)
    for name in terms:
        labels[hits[name] & (n_hits == 1)] = name
    labels[n_hits > 1] = both
    return labels


def topic_term_counts(
    texts: pd.Series,
    groups: pd.Series,
    terms: tuple[str, ...],
) -> pd.DataFrame:
    """Per term, in how many units of each group (language) it fires: a list that only
    fires in one language measures the list, not Louis."""
    normalised = texts.fillna("").map(normalise)
    counts = {t: normalised.str.contains(t).groupby(groups).sum() for t in terms}
    return pd.DataFrame(counts).T.rename_axis("term")


def neighbour_agreement(
    vectors: np.ndarray,
    labels: np.ndarray,
    k: int = 15,
) -> pd.DataFrame:
    """Does the encoder put units with the same topic near each other, in all dimensions?

    Per topic: of each unit's `k` nearest neighbours (cosine), the share with the same
    topic, against the share that topic has overall. A share well above the base rate:
    the position (encoder) and the colour (word lists) agree. Near it: they do not.
    """
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)  # a unit is not its own neighbour
    nearest = np.argpartition(-similarity, k, axis=1)[:, :k]
    same = labels[nearest] == labels[:, None]
    topics = np.unique(labels)
    return pd.DataFrame(
        {
            "units": [(labels == t).sum() for t in topics],
            "neighbours_same": [same[labels == t].mean() for t in topics],
            "base_rate": [(labels == t).mean() for t in topics],
        },
        index=pd.Index(topics, name="topic"),
    ).assign(lift=lambda d: d["neighbours_same"] / d["base_rate"])


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


class ContrastBarPlot(BasePlot):
    """One bar per cluster from 0: how many points more of his French than of his
    English falls in it (left: more in English). Sorted, so the clusters that lean
    furthest each way sit at the top and bottom.

    On each bar: the 95% interval (weeks resampled) as a line, the odd months as an open
    dot, the even months as a filled one. Bar, interval and both dots on the same side of
    0 is a difference that holds; dots on opposite sides is one that half the data does
    not show. Takes `contrast_intervals`'s frame.
    """

    def build(
        self,
        intervals: pd.DataFrame,
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
        ordered = intervals.sort_values("difference")
        y = np.arange(len(ordered))
        diff = ordered["difference"].to_numpy()
        self.ax.barh(
            y,
            diff,
            color=[colors[pair[0]] if d > 0 else colors[pair[1]] for d in diff],
            height=0.65,
            zorder=1,
        )
        self.ax.hlines(
            y,
            ordered["low"],
            ordered["high"],
            color=settings.DARK_GREY,
            linewidth=1.5,
            zorder=2,
        )
        self.ax.scatter(
            ordered["even_months"],
            y,
            color=settings.DARK_GREY,
            s=28,
            zorder=3,
            label="even months",
        )
        self.ax.scatter(
            ordered["odd_months"],
            y,
            facecolor="white",
            edgecolor=settings.DARK_GREY,
            linewidth=1.5,
            s=40,
            zorder=4,
            label="odd months",
        )
        self.ax.axvline(0, color=settings.GREY, linewidth=1)
        self.ax.set_yticks(y, [names.get(k, f"cluster {k}") for k in ordered.index])
        reach = np.nanmax(
            np.abs(ordered[["low", "high", "odd_months", "even_months"]].to_numpy())
        )
        self.ax.set_xlim(-reach * 1.1, reach * 1.1)
        self.ax.xaxis.set_major_formatter(lambda x, _: f"{x:+.0f} pts" if x else "0")
        self.ax.text(
            -0.04 * reach,
            len(ordered) - 0.3,
            f"← more in {pair_labels[pair[1]]}",
            ha="right",
            color="dimgray",
            fontsize=9,
        )
        self.ax.text(
            0.04 * reach,
            len(ordered) - 0.3,
            f"more in {pair_labels[pair[0]]} →",
            ha="left",
            color="dimgray",
            fontsize=9,
        )
        self.ax.set_ylim(-0.6, len(ordered) + 0.2)
        self.ax.grid(axis="x", linestyle=":", color=settings.LIGHT_GREY)
        self.ax.spines[["top", "right", "left"]].set_visible(False)
        self.ax.legend(loc="lower right", frameon=False)
        return self.fig, self.ax


CORNERS = {
    # corner: (x, y, ha, va) in axes fractions, and an arrow from that corner into the cloud
    "upper left": (0.02, 0.97, "left", "top", "↘"),
    "upper right": (0.98, 0.97, "right", "top", "↙"),
    "lower left": (0.02, 0.03, "left", "bottom", "↗"),
    "lower right": (0.98, 0.03, "right", "bottom", "↖"),
}


def corner_note(ax, corner: str, text: str, color: str, fontsize: int = 11) -> None:
    """Text in a corner of a map, with an arrow pointing from that corner into the cloud."""
    x, y, ha, va, arrow = CORNERS[corner]
    ax.text(
        x,
        y,
        f"{arrow} {text}",
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5},
    )


class ReportMapPanels(BasePlot):
    """One panel per language, built to carry one share: how much of each language is `focus`.

    All of a language's units in light grey, the `context` topics small and faded, the
    `focus` topic large on top, and the focus share as a big number in the panel. Shared
    axes (`extent`) and no tick labels: a projection's units mean nothing. Call with
    `n_plots=len(pair)`.
    """

    def build(
        self,
        coordinates: np.ndarray,
        topics: np.ndarray,
        languages: np.ndarray,
        focus: str = "love",
        context: tuple[str, ...] = ("planning",),
        pair: tuple[str, str] = ("fr", "en"),
        pair_labels: dict[str, str] | None = None,
        colors: dict[str, str] = settings.TOPIC_COLORS,
        extent: np.ndarray | None = None,
        notes: dict[str, tuple[str, str]] | None = None,
    ):
        if self.fig is None:
            raise ValueError("create_figure() must run before build()")
        axes = [ax for ax in self.fig.axes if ax.get_visible()]
        if len(axes) != len(pair):
            raise ValueError(f"{len(pair)} languages need n_plots={len(pair)}")
        pair_labels = pair_labels or {lang: lang for lang in pair}
        notes = notes or {}
        extent = coordinates if extent is None else extent
        pad = 0.03 * np.ptp(extent, axis=0)
        low, high = extent.min(axis=0) - pad, extent.max(axis=0) + pad
        for ax, lang in zip(axes, pair):
            mine = languages == lang
            ax.scatter(
                *coordinates[mine].T, s=7, color=settings.LIGHT_GREY, linewidth=0
            )
            for topic in context:
                hit = mine & (topics == topic)
                ax.scatter(
                    *coordinates[hit].T,
                    s=12,
                    color=colors[topic],
                    alpha=0.45,
                    linewidth=0,
                )
            hit = mine & (topics == focus)
            ax.scatter(
                *coordinates[hit].T, s=34, color=colors[focus], alpha=0.9, linewidth=0
            )
            ax.text(
                0.98,
                0.97,
                f"{hit.sum() / max(mine.sum(), 1):.0%}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=38,
                fontweight="bold",
                color=colors[focus],
            )
            for topic, (corner, text) in notes.items():
                corner_note(ax, corner, text, color=colors[topic])
            ax.set_title(
                f"{pair_labels[lang]}: {hit.sum():,} van {mine.sum():,} units",
                fontsize=12,
            )
            ax.set_xlim(low[0], high[0])
            ax.set_ylim(low[1], high[1])
            ax.set_xticks([])
            ax.set_yticks([])
        return self.fig, axes


class TopicMapPanels(BasePlot):
    """A projection as small multiples: one topic × one language per panel.

    Every panel shows all of that language's units in light grey and only one topic's
    units in colour on top, so a topic's place in the cloud is not hidden under the other
    topics or the other language. Rows are topics, columns languages; all panels share
    the same axes, so "lower right" means the same place everywhere. Call with
    `n_plots=len(topics) * len(pair)` and `max_cols=len(pair)`.
    """

    def build(
        self,
        coordinates: np.ndarray,
        topics: np.ndarray,
        languages: np.ndarray,
        show: tuple[str, ...] = ("love", "planning"),
        pair: tuple[str, str] = ("fr", "en"),
        names: dict[str, str] | None = None,
        pair_labels: dict[str, str] | None = None,
        colors: dict[str, str] = settings.TOPIC_COLORS,
        extent: np.ndarray | None = None,
        notes: dict[str, tuple[str, str]] | None = None,
    ):
        """`extent`: the coordinates the axis limits come from (default: `coordinates`). Pass
        the full set when drawing a subset, so two maps of different subsets line up.

        `notes`: per topic a (corner, text) pair, e.g. {"love": ("lower right", "vooral
        liefde")}, written in that corner of the topic's panels in the topic's colour. An
        observation about where the dots lean, not a name for the axis."""
        notes = notes or {}
        if self.fig is None:
            raise ValueError("create_figure() must run before build()")
        axes = [ax for ax in self.fig.axes if ax.get_visible()]
        if len(axes) != len(show) * len(pair):
            raise ValueError(
                f"{len(show)} topics × {len(pair)} languages need n_plots={len(show) * len(pair)}"
            )
        names = names or {t: t for t in show}
        pair_labels = pair_labels or {lang: lang for lang in pair}
        extent = coordinates if extent is None else extent
        pad = 0.03 * np.ptp(extent, axis=0)
        low, high = extent.min(axis=0) - pad, extent.max(axis=0) + pad
        for ax, (topic, lang) in zip(axes, [(t, lang) for t in show for lang in pair]):
            mine = languages == lang
            hit = mine & (topics == topic)
            ax.scatter(
                *coordinates[mine].T, s=7, color=settings.LIGHT_GREY, linewidth=0
            )
            ax.scatter(
                *coordinates[hit].T, s=24, color=colors[topic], alpha=0.85, linewidth=0
            )
            if topic in notes:
                corner_note(ax, *notes[topic], color=colors[topic])
            ax.set_title(
                f"{names[topic]}, {pair_labels[lang]}: {hit.sum():,} van {mine.sum():,} units "
                f"({hit.sum() / max(mine.sum(), 1):.0%})",
                fontsize=10,
            )
            ax.set_xlim(low[0], high[0])
            ax.set_ylim(low[1], high[1])
            ax.set_xticks([])
            ax.set_yticks([])
        return self.fig, axes


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

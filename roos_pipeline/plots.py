"""Roos's own plot classes: every chart the pipeline draws, as a `BasePlot` subclass.

Moved here on 2026-10-06 from `scripts/plots.py`, which is the teacher's file: the
first seven classes below (`RiseDeclinePlot` to `EventMarker`) were written for the
roos_03.3 and roos_04.6 notebooks and are unchanged. The last four are new, one per
pipeline figure that had no class yet. `HomeAwayPlot` (figure 7) and `StyleMapPlot`
(figure 8) came last, from roos_07.4 and roos_06.6 (2026-11-13).

Why subclass `BasePlot` instead of six lines of `plt.subplots` in a function:
`PlotSettings` carries the styling (figure size, labels, grey/highlight colours), so a
chart can be restyled without touching its drawing code, and `plot_on` stacks one
class on top of another (an `EventMarker` on a `LinePlot`) on the same axes.

Every class draws only. The numbers it draws (shares, fitted curves, p-values) are
computed in `aggregate.py` and passed in.
"""

from __future__ import annotations

import textwrap

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from goad_toolkit.visualizer import BasePlot, LinePlot
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy import stats

from roos_pipeline import settings
from scripts.plots import BarPlot


class RiseDeclinePlot(BasePlot):
    """A line chart whose segments are coloured by direction, not by category.

    Neither `LinePlot` (goad_toolkit) nor `BarPlot` above say anything about whether a
    series is going up or down at a given point — that is exactly what "the rise and
    decline of X" asks for, so it gets its own class rather than a kwarg bolted onto an
    existing one.

    Two colours carry the whole story: `settings.highlight_color` for a segment that
    rose since the previous point, `settings.base_color` for one that fell or held
    steady — grey-first, per 02.2's rule, since a climb is the finding and a flat or
    falling stretch is the backdrop for it. A run of equal values keeps the direction
    of the segment before it, so a plateau at a peak doesn't flip colour for no reason.
    The legend gets exactly two entries (rise/decline), not one per segment.
    """

    def build(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        rise_color: str | None = None,
        decline_color: str | None = None,
        rise_label: str = "rising",
        decline_label: str = "declining",
        marker: str = "o",
        **kwargs,
    ):
        if self.ax is None:
            raise ValueError("create_figure() must run before build()")

        ordered = data.sort_values(x)
        xs = ordered[x].to_numpy()
        ys = ordered[y].to_numpy()
        if len(xs) < 2:
            raise ValueError(
                "RiseDeclinePlot needs at least two points to have a direction"
            )

        rise_color = rise_color or self.settings.highlight_color
        decline_color = decline_color or self.settings.base_color

        rising = True
        for i in range(1, len(xs)):
            if ys[i] != ys[i - 1]:
                rising = ys[i] > ys[i - 1]
            color = rise_color if rising else decline_color
            self.ax.plot(
                xs[i - 1 : i + 1],
                ys[i - 1 : i + 1],
                color=color,
                marker=marker,
                **kwargs,
            )

        self.ax.legend(
            handles=[
                Line2D([0], [0], color=rise_color, marker=marker, label=rise_label),
                Line2D(
                    [0], [0], color=decline_color, marker=marker, label=decline_label
                ),
            ]
        )
        return self.fig, self.ax


class DistributionDiffPlot(BasePlot):
    """How much more (or less) likely each value became, not the two distributions
    themselves.

    Takes the *difference* between an "after" and a "before" density/mass at the same
    x -- the residual-over-time move (03.1, `SubtractBaseline`) applied to two fitted
    shapes instead of two time periods. Positive: more likely after. Negative: less
    likely after. Built for laying two people's (or two groups') shifts on one shared
    axis with a zero line, when the two *directions* of change are the finding rather
    than either distribution's shape on its own.

        x = np.arange(1, 12)
        diff = after_fit.frozen_dist.pmf(x) - before_fit.frozen_dist.pmf(x)
        DistributionDiffPlot(settings).plot(x=x, diff=diff, label="Roos")

    `x`/`diff` are computed in the notebook, not here -- this class only draws them,
    same division of labour as `BarPlot` above.
    """

    def build(
        self,
        x,
        diff,
        color: str | None = None,
        label: str | None = None,
        fill: bool = True,
        **kwargs,
    ):
        if self.ax is None:
            raise ValueError("create_figure() must run before build()")
        color = color or self.settings.highlight_color
        self.ax.plot(x, diff, color=color, marker="o", label=label, **kwargs)
        if fill:
            self.ax.fill_between(x, diff, 0, color=color, alpha=0.15)
        return self.fig, self.ax


class RidgelinePlot(BasePlot):
    """One density curve per group, stacked vertically and allowed to overlap, so many
    groups' *shapes* share one x-axis without a tangle of overlaid lines.

    Each ridge is a KDE scaled to the same peak height -- the ridges compare shape and
    position, not group size, so a 1,000-message group and a 6,000-message one draw
    equally tall. Say the n's somewhere else (the labels are a good place).

    `log_scale=True` fits the KDE on log10 of the values and draws it on a log x-axis,
    which is what a long right tail (message length, response time) needs -- on a
    linear axis everything piles up against zero. A small tick marks each group's
    median, so "shifted right" reads as a number, not just an impression.

    Colour follows 02.2's grey-first rule: every ridge is `settings.base_color` unless
    it's in `highlight_groups`, which gets `settings.highlight_color`.

        RidgelinePlot(settings).plot(data=frame, x="n_chars", group="category",
                                     order=[...], highlight_groups=["planning_logistics"])
    """

    def build(
        self,
        data: pd.DataFrame,
        x: str,
        group: str,
        order: list[str],
        highlight_groups: list[str] | None = None,
        labels: dict[str, str] | None = None,
        log_scale: bool = True,
        overlap: float = 1.6,
        bw_adjust: float = 1.0,
        n_grid: int = 400,
    ):
        if self.ax is None:
            raise ValueError("create_figure() must run before build()")
        highlight_groups = highlight_groups or []
        labels = labels or {}

        transform = np.log10 if log_scale else (lambda v: v)
        values_all = transform(data[x].to_numpy(dtype=float))
        grid = np.linspace(values_all.min(), values_all.max(), n_grid)
        grid_x = 10**grid if log_scale else grid

        # First group on top: its baseline sits highest, each next one a step lower.
        for i, name in enumerate(order):
            baseline = len(order) - 1 - i
            values = transform(data.loc[data[group] == name, x].to_numpy(dtype=float))
            # set bandwidth when create the KDE to avoid ty error unsupported operator
            # kde.factor in scipy's type hints are loose
            # they allow bytes, memoryview, as numbers
            kde = stats.gaussian_kde(
                values, bw_method=lambda k: k.scotts_factor() * bw_adjust
            )
            density = kde(grid)
            height = density / density.max() * overlap
            color = (
                self.settings.highlight_color
                if name in highlight_groups
                else self.settings.base_color
            )
            # zorder: lower ridges drawn over the ones above, so each outline stays
            # visible where they overlap.
            z = 2 * i
            self.ax.fill_between(
                grid_x, baseline, baseline + height, color=color, zorder=z
            )
            self.ax.plot(
                grid_x, baseline + height, color="dimgray", linewidth=0.8, zorder=z + 1
            )
            self.ax.axhline(baseline, color="dimgray", linewidth=0.5, zorder=z + 1)

            median = np.median(values)
            median_height = kde(median)[0] / density.max() * overlap
            median_x = 10**median if log_scale else median
            self.ax.plot(
                [median_x, median_x],
                [baseline, baseline + median_height],
                color="black",
                linewidth=1.2,
                zorder=z + 1,
            )

        if log_scale:
            self.ax.set_xscale("log")
        self.ax.set_xlim(grid_x.min(), grid_x.max())
        self.ax.set_yticks(range(len(order) - 1, -1, -1))
        self.ax.set_yticklabels([labels.get(name, name) for name in order])
        self.ax.set_ylim(-0.1, len(order) - 1 + overlap + 0.3)
        self.ax.spines[["top", "right", "left"]].set_visible(False)
        self.ax.tick_params(axis="y", length=0)
        return self.fig, self.ax


class ViolinPlot(BasePlot):
    """Horizontal violins, one per group: a mirrored density with the quartiles drawn
    inside, so shape *and* summary (median, IQR) sit in the same mark.

    Same colour rule and log-scale reasoning as `RidgelinePlot` above. `cut=0` stops
    the density at the observed min/max rather than inventing values beyond them --
    a message can't be shorter than one character. `density_norm="width"` gives every
    violin the same maximum width, again comparing shape rather than group size.
    """

    def build(
        self,
        data: pd.DataFrame,
        x: str,
        group: str,
        order: list[str],
        highlight_groups: list[str] | None = None,
        labels: dict[str, str] | None = None,
        log_scale: bool = True,
        **kwargs,
    ):
        if self.ax is None:
            raise ValueError("create_figure() must run before build()")
        highlight_groups = highlight_groups or []
        labels = labels or {}
        kwargs.setdefault(
            "palette",
            {
                name: self.settings.highlight_color
                if name in highlight_groups
                else self.settings.base_color
                for name in order
            },
        )
        kwargs.setdefault("inner", "quart")
        kwargs.setdefault("cut", 0)
        kwargs.setdefault("density_norm", "width")
        sns.violinplot(
            data=data,
            x=x,
            y=group,
            hue=group,
            order=order,
            hue_order=order,
            log_scale=log_scale,
            legend=False,
            ax=self.ax,
            **kwargs,
        )
        self.ax.set_yticks(
            range(len(order)), [labels.get(name, name) for name in order]
        )
        # seaborn writes the column name onto the axis; the settings get the last word.
        if self.settings.ylabel is not None:
            self.ax.set_ylabel(self.settings.ylabel)
        return self.fig, self.ax


class BeforeAfterDensityPlot(BasePlot):
    """Two samples as overlaid density histograms on identical bins, each with its fitted
    curve drawn over it in the same colour, and a direct label instead of a legend.

    The point is that bars and curve are on the *same* density scale, so a reader can see
    where the fit misses:

    - `log_x=False`: bar height = share of observations per unit of x (`density=True`).
      For a count, pass the pmf as `density`; with integer-width bins that is the same
      scale.
    - `log_x=True`: `bins` are log-spaced edges in the original unit. Bar height = share
      of observations per unit of log10(x), and the curve is converted to that scale too
      (`f(x) * x * ln 10`), so a lognormal draws as a bell and the bars line up with it.

        BeforeAfterDensityPlot(settings).plot(
            samples={"before": a, "after": b}, bins=edges, colors=SIDE_COLORS,
            densities={"before": fit_a.pdf, "after": fit_b.pdf},
            labels={"before": "before: μ = 97", "after": "after: μ = 103"},
            label_positions={"before": (0.62, 0.85), "after": (0.62, 0.72)},
        )

    `densities` and the label text are computed in the notebook; this class only draws.
    `label_positions` are in axes fractions (0-1), so they land in the same place
    whatever the data's scale.
    """

    def build(
        self,
        samples: dict[str, np.ndarray],
        bins: np.ndarray,
        colors: dict[str, str],
        densities: dict | None = None,
        labels: dict[str, str] | None = None,
        label_positions: dict[str, tuple[float, float]] | None = None,
        log_x: bool = False,
        discrete: bool = False,
        alpha: float = 0.25,
    ):
        if self.ax is None:
            raise ValueError("create_figure() must run before build()")
        densities = densities or {}
        labels = labels or {}
        label_positions = label_positions or {}
        edges = np.asarray(bins, dtype=float)

        for name, values in samples.items():
            values = np.asarray(values, dtype=float)
            color = colors[name]
            if log_x:
                heights, _ = np.histogram(
                    np.log10(values), bins=np.log10(edges), density=True
                )
            else:
                heights, _ = np.histogram(values, bins=edges, density=True)
            if discrete and not log_x:
                # One bar per bin, with gaps: two neighbouring counts with the same height
                # must not merge into what looks like one wide bin.
                centers, widths = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
                self.ax.bar(
                    centers,
                    heights,
                    width=widths * 0.85,
                    color=color,
                    alpha=alpha + 0.15,
                    edgecolor=color,
                    linewidth=1,
                )
            else:
                self.ax.stairs(heights, edges, fill=True, color=color, alpha=alpha)
                self.ax.stairs(heights, edges, color=color, linewidth=1)

            density = densities.get(name)
            if density is not None:
                if log_x:
                    grid = np.geomspace(edges[0], edges[-1], 400)
                    curve = density(grid) * grid * np.log(10)
                elif discrete:
                    grid = np.arange(np.ceil(edges[0]), np.floor(edges[-1]) + 1)
                    curve = density(grid)
                else:
                    grid = np.linspace(edges[0], edges[-1], 400)
                    curve = density(grid)
                self.ax.plot(grid, curve, color=color, linewidth=2.5)

            if name in labels:
                x, y = label_positions.get(name, (0.98, 0.9))
                self.ax.text(
                    x,
                    y,
                    labels[name],
                    transform=self.ax.transAxes,
                    color=color,
                    fontsize=10,
                    fontweight="bold",
                    ha="left",
                    va="top",
                )

        if log_x:
            self.ax.set_xscale("log")
        self.ax.set_xlim(edges[0], edges[-1])
        self.ax.spines[["top", "right"]].set_visible(False)
        return self.fig, self.ax


class DotStripPlot(BasePlot):
    """One dot per observation, one column per group, with a short line at each group's
    summary value: every single week stays visible, which matters when a group is small
    (10 weeks) and a box or a mean would hide how many of them actually stand out.

    Grey-first (02.2): every group is `settings.base_color` unless it's in
    `highlight_groups`. `summaries` maps group -> value for the line, `summary_labels`
    group -> the text written next to it, so the number sits on the chart, not in a legend.

        DotStripPlot(settings).plot(data=weekly, x="group", y="share", order=[...],
                                    highlight_groups=["hard weeks"],
                                    summaries={"hard weeks": 22.6}, summary_labels={...})
    """

    def build(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        order: list[str],
        highlight_groups: list[str] | None = None,
        summaries: dict[str, float] | None = None,
        summary_labels: dict[str, str] | None = None,
        tick_labels: dict[str, str] | None = None,
        jitter: float = 0.18,
        seed: int = 0,
    ):
        if self.ax is None:
            raise ValueError("create_figure() must run before build()")
        highlight_groups = highlight_groups or []
        summaries = summaries or {}
        summary_labels = summary_labels or {}
        tick_labels = tick_labels or {}
        rng = np.random.default_rng(seed)

        for i, group in enumerate(order):
            color = (
                self.settings.highlight_color
                if group in highlight_groups
                else self.settings.base_color
            )
            values = data.loc[data[x] == group, y].to_numpy(dtype=float)
            offsets = rng.uniform(-jitter, jitter, len(values))
            self.ax.scatter(
                i + offsets,
                values,
                color=color,
                s=40,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )
            if group in summaries:
                self.ax.hlines(
                    summaries[group],
                    i - 0.32,
                    i + 0.32,
                    color="black",
                    linewidth=2.5,
                    zorder=3,
                )
            if group in summary_labels:
                self.ax.text(
                    i + 0.36,
                    summaries.get(group, 0),
                    summary_labels[group],
                    va="center",
                    ha="left",
                    fontsize=9.5,
                    fontweight="bold",
                    color="black" if group not in highlight_groups else color,
                )

        self.ax.set_xticks(range(len(order)), [tick_labels.get(g, g) for g in order])
        self.ax.set_xlim(-0.6, len(order) - 0.1)
        self.ax.spines[["top", "right"]].set_visible(False)
        return self.fig, self.ax


class EventMarker(BasePlot):
    """A quiet vertical line at one date, with an arrow and a label pointing at it.

    `VerticalDate` is a loud red dashed line that lands in the legend. This one is grey
    (so it recedes behind the data, per 02.2's grey-first rule) and carries its own
    label next to the line, so the legend does not need an entry for it.

    Drawn on top of an existing chart with `plot_on`:

        lines.plot_on(EventMarker(settings), date="2025-05-11", label="baby is born")

    `height` is where the label sits, as a fraction of the axes height (0 = bottom,
    1 = top), so it lands in the same place whatever the y-scale is. `text_offset_days`
    is how far from the line the label starts, in days: negative puts the label to the
    left of the line, positive to the right.
    """

    def build(
        self,
        date: str,
        label: str,
        height: float = 0.9,
        text_offset_days: int = -30,
        color: str = "grey",
    ):
        if self.ax is None:
            raise ValueError("create_figure() must run before build()")
        # Numeric x, so the marker lands on the right day whether or not the axis
        # carries timezone-aware timestamps.
        x = mdates.date2num(pd.to_datetime(date))
        self.ax.axvline(x, color=color, linestyle="--", linewidth=1.5, zorder=1)
        self.ax.annotate(
            label,
            xy=(x, height),
            xytext=(x + text_offset_days, height),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            ha="right" if text_offset_days < 0 else "left",
            va="center",
            color="dimgray",
            arrowprops={"arrowstyle": "->", "color": color},
        )
        return self.fig, self.ax


# --- new for the pipeline (2026-10-06) ---------------------------------------------


class PeriodPanelsPlot(BasePlot):
    """One bar panel per period, the same categories in the same order in each.

    The roos_01.4 "Praat Nederlands met mij" layout: before and after the move side by
    side, sharing a y-axis. Call with `n_plots=` the number of panels, and put the
    panel titles in `PlotSettings.subplot_titles`.

    Colour comes from the settings, grey-first (02.2): a bar is
    `settings.highlight_color` if its category is in `settings.highlight`, otherwise
    `settings.base_color`. So "which language is the story" is a settings choice, not
    a colour dict passed around.
    """

    def build(
        self,
        data: pd.DataFrame,
        panel: str,
        x: str,
        y: str,
        panels: list[str],
        order: list[str],
        fmt: str = "{:.0f}%",
        ylim: tuple[float, float] = (0, 100),
    ):
        if self.fig is None:
            raise ValueError("create_figure() must run before build()")
        axes = [ax for ax in self.fig.axes if ax.get_visible()]
        if len(axes) != len(panels):
            raise ValueError(
                f"{len(panels)} panels need plot(n_plots={len(panels)}, ...), "
                f"got {len(axes)} axes"
            )
        for ax, value in zip(axes, panels):
            subset = data[data[panel] == value].set_index(x)[y].reindex(order).fillna(0)
            colors = [
                self.settings.highlight_color
                if category in self.settings.highlight
                else self.settings.base_color
                for category in order
            ]
            bars = ax.bar(order, subset.to_numpy(), color=colors)
            ax.bar_label(bars, labels=[fmt.format(v) for v in subset], padding=3)
            ax.set_ylim(*ylim)
            ax.spines[["top", "right"]].set_visible(False)
        return self.fig, axes


class LabelledBarPlot(BarPlot):
    """The teacher's `BarPlot` (02.2), plus the value written on every bar.

    Inherits rather than copies: `BarPlot.build` still draws the bars (seaborn, hue,
    order, palette), this class only adds the labels afterwards. If 02.2's `BarPlot`
    changes, this one follows.
    """

    def build(self, data: pd.DataFrame, x: str, y: str, fmt: str = "%.0f%%", **kwargs):
        fig, ax = super().build(data=data, x=x, y=y, **kwargs)
        for container in ax.containers:
            ax.bar_label(container, fmt=fmt, padding=2, fontsize=9)
        return fig, ax


class TopicTrendPlot(BasePlot):
    """Weekly topic shares: grey raw weeks, a coloured rolling average per topic, and a
    grey `EventMarker` per event. The roos_03.3 "Baby is taking over" chart.

    Built entirely from other `BasePlot` classes stacked with `plot_on`: goad's
    `LinePlot` (with `gaps=True`, so an empty week breaks the line) for every line, and
    `EventMarker` for every event. `weekly` holds the raw shares, `trend` the rolling
    averages as `<topic>_rolling_avg` (what goad's `RollingAvg(rename=True)` writes).
    """

    def build(
        self,
        weekly: pd.DataFrame,
        trend: pd.DataFrame,
        x: str,
        colors: dict[str, str],
        events: dict[str, str],
        smooth_weeks: int,
        raw_color: str = "lightgray",
    ):
        if self.ax is None:
            raise ValueError("create_figure() must run before build()")
        # The real weekly values recede to grey; the rolling average carries the colour.
        for topic in colors:
            self.plot_on(
                LinePlot(self.settings),
                data=weekly,
                gaps=True,
                x=x,
                y=topic,
                color=raw_color,
                linewidth=1,
            )
        for topic, color in colors.items():
            self.plot_on(
                LinePlot(self.settings),
                data=trend,
                gaps=True,
                x=x,
                y=f"{topic}_rolling_avg",
                color=color,
                linewidth=2.5,
                label=f"{topic} ({smooth_weeks}-week average)",
            )
        for label, day in events.items():
            self.plot_on(EventMarker(self.settings), date=day, label=label)
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        return self.fig, self.ax


class PeakLabelledCurves(BasePlot):
    """Several curves on one axis, each labelled at its own peak instead of a legend.

    The roos_04.6d "prepare and plan" chart: two fitted negative binomials, normal
    weeks and hard weeks, so the gap between them shows as two peaks far apart. Every
    curve is a goad `LinePlot`; this class only adds the direct labels.

        PeakLabelledCurves(settings).plot(data=curves, x="k",
                                          curves={"normal": ("grey", "normal weeks")})
    """

    def build(
        self,
        data: pd.DataFrame,
        x: str,
        curves: dict[str, tuple[str, str]],
        label_offset: float = 1.5,
        linewidth: float = 2.5,
    ):
        if self.ax is None:
            raise ValueError("create_figure() must run before build()")
        for column, (color, label) in curves.items():
            self.plot_on(
                LinePlot(self.settings),
                data=data,
                x=x,
                y=column,
                gaps=True,
                color=color,
                linewidth=linewidth,
            )
            # nanargmax, like idxmax: a NaN is never the peak
            peak = data.iloc[int(np.nanargmax(data[column].to_numpy(dtype=float)))]
            self.ax.text(
                peak[x] + label_offset,
                peak[column],
                label,
                color=color,
                fontweight="bold",
                va="center",
            )
        self.ax.set_ylim(0, None)
        self.ax.spines[["top", "right"]].set_visible(False)
        return self.fig, self.ax


# The light outline that keeps a label readable on top of a line.
HALO = [patheffects.withStroke(linewidth=4, foreground=settings.PAPER)]


class HomeAwayPlot(BasePlot):
    """On-the-road themes in a column left, home themes right, places in between.

    A place sits at x = its thuis-score, sorted from most on the road (top) to most home
    (bottom). A line joins a place to a theme; its colour is the theme's group, its width
    and opacity grow with the share, and shares below `min_share` are not drawn. Only the
    places in `labelled` get their percentages, so the two extremes can be read without
    covering the picture in numbers. All text is black or grey, never the line colour.
    `caption` is printed under the plot; a blank line in it starts a new paragraph.
    """

    def build(
        self,
        table: pd.DataFrame,
        subtitle: str,
        themes: dict[str, tuple[str, str]] = settings.HOME_THEMES,
        labelled: tuple[str, ...] = ("Parijs", "Nieuwegein"),
        min_share: float = settings.HOME_MIN_SHARE,
        max_width: float = 16.0,
        caption: str | None = None,
        caption_width: int = 175,
    ):
        if self.ax is None or self.fig is None:
            raise ValueError("create_figure() must run before build()")
        ax, fig = self.ax, self.fig
        fig.set_facecolor(settings.PAPER)
        ax.set_facecolor(settings.PAPER)
        color = {settings.AWAY: settings.AWAY_COLOR, settings.HOME: settings.HOME_COLOR}

        order = table.sort_values("thuis_score").index
        top = len(order) - 1
        place_pos = {
            p: (table.loc[p, "thuis_score"], top - i) for i, p in enumerate(order)
        }
        lo, hi = table["thuis_score"].min(), table["thuis_score"].max()
        spread = hi - lo
        left, right = lo - 0.55 * spread, hi + 0.55 * spread
        theme_pos = {}
        for group, x in ((settings.AWAY, left), (settings.HOME, right)):
            mine = [t for t, (g, _) in themes.items() if g == group]
            for t, y in zip(mine, np.linspace(top - 0.3, 0.3, len(mine))):
                theme_pos[t] = (x, y)

        heaviest = table[list(themes)].to_numpy().max()
        for place, (px, py) in place_pos.items():
            for theme, (group, _) in themes.items():
                share = table.loc[place, theme]
                if share < min_share:
                    continue
                tx, ty = theme_pos[theme]
                ax.plot(
                    [px, tx],
                    [py, ty],
                    color=color[group],
                    linewidth=0.8 + (max_width - 0.8) * share / heaviest,
                    alpha=0.15 + 0.6 * share / heaviest,
                    solid_capstyle="round",
                    zorder=1,
                )
                if place in labelled:
                    # a steep line crowds the others near the place: put its number near the theme
                    steep = abs(ty - py) / (abs(tx - px) / spread) > 2.2
                    t = 0.78 if steep else 0.5
                    ax.text(
                        px + t * (tx - px),
                        py + t * (ty - py),
                        f"{share:.0%}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=settings.DARK_GREY,
                        path_effects=HALO,
                        zorder=4,
                    )

        paper = {
            "boxstyle": "round,pad=0.25",
            "facecolor": settings.PAPER,
            "edgecolor": "none",
            "alpha": 0.9,
        }
        for place, (px, py) in place_pos.items():
            ax.scatter(
                px,
                py,
                s=160,
                color=settings.DARK_GREY,
                edgecolors=settings.PAPER,
                linewidths=1.5,
                zorder=3,
            )
            ax.annotate(
                f"{table.loc[place, 'berichten']} berichten",
                (px, py),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=settings.GREY,
                bbox=paper,
                zorder=5,
            )
            ax.annotate(
                place,
                (px, py),
                xytext=(0, 27),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12.5,
                fontweight="bold",
                color="black",
                bbox=paper,
                zorder=5,
            )

        for theme, (x, y) in theme_pos.items():
            group = themes[theme][0]
            ax.text(
                x,
                y,
                theme,
                ha="right" if group == settings.AWAY else "left",
                va="center",
                fontsize=12.5,
                color="black",
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "facecolor": settings.PAPER,
                    "edgecolor": color[group],
                    "linewidth": 2.2,
                },
            )
        for x, header, ha in (
            (left, "SPONTAAN OP PAD", "right"),
            (right, "THUIS MET KIND", "left"),
        ):
            ax.text(
                x,
                top + 0.45,
                header,
                ha=ha,
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=settings.GREY,
            )

        ax.annotate(
            "",
            xy=(hi + 0.1 * spread, -0.75),
            xytext=(lo - 0.1 * spread, -0.75),
            arrowprops={"arrowstyle": "-|>", "color": settings.GREY, "linewidth": 1.4},
        )
        ax.text(
            (lo + hi) / 2,
            -0.9,
            "hoe verder naar rechts, hoe meer de gesprekken gaan over thuis met kind",
            ha="center",
            va="top",
            fontsize=10.5,
            color=settings.GREY,
        )

        ax.set_title(
            ""
        )  # BasePlot's centred title: this one sits on the left, above the subtitle
        ax.set_title(
            self.settings.title,
            loc="left",
            fontsize=20,
            fontweight="bold",
            color="black",
            pad=40 + 21 * subtitle.count("\n"),
        )
        ax.text(
            0,
            1.03,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            color=settings.GREY,
            linespacing=1.4,
        )
        if caption is not None:
            # one paragraph per "\n\n", each wrapped on its own
            wrapped = "\n\n".join(
                textwrap.fill(part, caption_width) for part in caption.split("\n\n")
            )
            ax.text(
                0,
                -0.01,
                wrapped,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10.5,
                color=settings.DARK_GREY,
                linespacing=1.45,
            )
        ax.set_xlim(left - 0.32 * spread, right + 0.32 * spread)
        ax.set_ylim(-1.3, top + 0.8)
        ax.axis("off")
        return fig, ax


def shade(color: str, t: float) -> tuple[float, float, float]:
    """The person's colour from light (t = 0, 2023) to full (t = 1, 2026)."""
    weight = 0.25 + 0.75 * float(np.clip(t, 0, 1))
    r, g, b = to_rgb(color)
    return 1 - weight * (1 - r), 1 - weight * (1 - g), 1 - weight * (1 - b)


class StyleMapPlot(BasePlot):
    """Figure 8: documents as small dots, half-year means as paths with arrows.

    `docs` has one row per document (`author`, `ts_local`, `x`, `y`), `means` one row
    per person per half-year. Colour is the person, lightness is time (light 2023,
    full 2026). `texts` holds every sentence on it (title, subtitle, the words at the
    ends of the axes and their examples, footnote), so the wording is decided in
    `figures.py`, not here. Moved from scripts/stijlkaart.py (`draw_stijlkaart`).
    """

    def build(self, docs: pd.DataFrame, means: pd.DataFrame, texts: dict[str, str]):
        if self.ax is None or self.fig is None:
            raise ValueError("create_figure() must run before build()")
        fig, ax = self.fig, self.ax
        start, end = docs["ts_local"].min(), docs["ts_local"].max()

        def time_share(when: pd.Series) -> np.ndarray:
            return ((when - start) / (end - start)).to_numpy(dtype=float)

        fig.set_facecolor(settings.PAPER)
        ax.set_facecolor(settings.PAPER)
        ax.set_title("")
        ax.set_position((0.1, 0.2, 0.85, 0.62))  # room for the texts around it
        for spine in ax.spines.values():
            spine.set_color(settings.STYLE_FRAME)
        ax.set_xticks([])
        ax.set_yticks([])

        for author, color in settings.STYLE_PERSON_COLORS.items():
            mine = (docs["author"] == author).to_numpy()
            ax.scatter(
                docs.loc[mine, "x"],
                docs.loc[mine, "y"],
                s=16,
                alpha=0.35,
                linewidths=0,
                color=[shade(color, t) for t in time_share(docs.loc[mine, "ts_local"])],
            )

        x_range = np.ptp(docs["x"])
        span = np.ptp(docs[["x", "y"]].to_numpy(), axis=0)
        for author, color in settings.STYLE_PERSON_COLORS.items():
            path = (
                means[means["author"] == author]
                .sort_values("ts_local")
                .reset_index(drop=True)
            )
            t = time_share(path["ts_local"])
            points = path[["x", "y"]].to_numpy()
            for i in range(1, len(points)):
                if (
                    np.linalg.norm(points[i] - points[i - 1])
                    < settings.STYLE_MIN_ARROW_SHARE * x_range
                ):
                    continue
                ax.add_patch(
                    FancyArrowPatch(
                        points[i - 1],
                        points[i],
                        arrowstyle="-|>",
                        mutation_scale=18,
                        linewidth=3,
                        color=shade(color, t[i]),
                        shrinkA=6,
                        shrinkB=6,
                        zorder=3,
                    )
                )
            sizes = np.full(len(points), 40.0)
            sizes[[0, -1]] = 110.0
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=sizes,
                color=[shade(color, ti) for ti in t],
                zorder=4,
                linewidths=0,
            )
            for idx, neighbour, year in ((0, 1, start.year), (-1, -2, end.year)):
                # away from the trajectory, measured in axis widths so x and y count the same
                away = (points[idx] - points[neighbour]) / span
                away = away / (np.linalg.norm(away) + 1e-12)
                ax.annotate(
                    f"{author} {year}",
                    points[idx],
                    xytext=(34 * away[0], 34 * away[1]),
                    textcoords="offset points",
                    ha="left" if away[0] >= 0 else "right",
                    va="bottom" if away[1] >= 0 else "top",
                    fontsize=13,
                    fontweight="bold",
                    color=color,
                    zorder=5,
                    bbox={
                        "facecolor": settings.PAPER,
                        "edgecolor": "none",
                        "alpha": 0.85,
                        "pad": 1.5,
                    },
                )

        ax.text(
            0.985,
            0.025,
            "grote stip = gemiddelde per halfjaar · kleine stippen = steekproeven van "
            "±600 woorden\nlicht = 2023, donker = 2026",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            color=settings.STYLE_MUTED,
        )

        # axis ends: a bold word, grey examples underneath
        ax.text(
            0,
            -0.035,
            texts["x_low"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=settings.STYLE_TEXT,
        )
        ax.text(
            0,
            -0.085,
            texts["x_low_examples"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            color=settings.STYLE_MUTED,
        )
        ax.text(
            1,
            -0.035,
            texts["x_high"],
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=settings.STYLE_TEXT,
        )
        ax.text(
            1,
            -0.085,
            texts["x_high_examples"],
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            color=settings.STYLE_MUTED,
        )
        ax.text(
            -0.06,
            0,
            texts["y_low"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=13,
            fontweight="bold",
            color=settings.STYLE_TEXT,
        )
        ax.text(
            -0.03,
            0,
            texts["y_low_examples"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=10.5,
            color=settings.STYLE_MUTED,
        )
        ax.text(
            -0.06,
            1,
            texts["y_high"],
            transform=ax.transAxes,
            ha="center",
            va="top",
            rotation=90,
            fontsize=13,
            fontweight="bold",
            color=settings.STYLE_TEXT,
        )
        ax.text(
            -0.03,
            1,
            texts["y_high_examples"],
            transform=ax.transAxes,
            ha="center",
            va="top",
            rotation=90,
            fontsize=10.5,
            color=settings.STYLE_MUTED,
        )

        fig.text(
            0.1,
            0.955,
            texts["title"],
            ha="left",
            va="top",
            fontsize=24,
            fontweight="bold",
            color=settings.STYLE_TEXT,
        )
        fig.text(
            0.1,
            0.895,
            texts["subtitle"],
            ha="left",
            va="top",
            fontsize=13,
            color=settings.STYLE_TEXT,
            linespacing=1.6,
            alpha=0.8,
        )
        fig.text(
            0.1,
            0.105,
            texts["footnote"],
            ha="left",
            va="top",
            fontsize=8.5,
            color=settings.STYLE_MUTED,
            linespacing=1.6,
        )
        return fig, ax

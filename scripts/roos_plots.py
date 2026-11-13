"""Roos's plot classes for the notebooks of lessons 3 and 4, as `BasePlot` subclasses.

Written for roos_03.3 and roos_04.6 (2026-09/10), moved from `scripts/plots.py` to
`roos_pipeline/plots.py` on 2026-10-06, and here on 2026-11-13: no pipeline figure uses
them, and `roos_pipeline/` holds only the code of the six final figures. `EventMarker`
stayed in `roos_pipeline.plots`, because les 3's `TopicTrendPlot` is built from it.

Why subclass `BasePlot` instead of six lines of `plt.subplots` in a function:
`PlotSettings` carries the styling (figure size, labels, grey/highlight colours), so a
chart can be restyled without touching its drawing code, and `plot_on` stacks one
class on top of another (an `EventMarker` on a `LinePlot`) on the same axes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from goad_toolkit.visualizer import BasePlot
from matplotlib.lines import Line2D
from scipy import stats


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

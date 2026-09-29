"""The plot classes lesson 2 derives, in a form later lessons and dashboards can import.

Lesson 2 writes both of these from scratch — that derivation is the exercise and it stays
in the notebook. This module is where the same code lives afterwards, so a dashboard can
render the chart without a copy of it drifting out of sync with the notebook's version.

    from scripts.plots import BarPlot

    fig, ax = BarPlot(settings).plot(data=frame, x="author", y="degree")

That import is the whole point of subclassing `BasePlot` rather than writing six lines of
`plt.subplots` in a cell: there is no way to import a chart that only exists as a cell.
`fig` is an ordinary matplotlib figure, so `st.pyplot(fig)` takes it as-is and the class
never learns that streamlit exists.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from goad_toolkit.visualizer import BasePlot
from matplotlib.lines import Line2D
from scipy import stats


class BarPlot(BasePlot):
    """A bar chart. All the styling lives in PlotSettings."""

    def build(self, data: pd.DataFrame, x: str, y: str, **kwargs):
        sns.barplot(data=data, x=x, y=y, ax=self.ax, **kwargs)
        return self.fig, self.ax


class BarPlotWithError(BasePlot):
    """A grouped bar chart carrying intervals that were computed elsewhere.

    `GroupedBarPlot` cannot do this: seaborn derives its intervals from raw
    observations, and these are already-summarised means with intervals of their own.

    Each bar is looked up by (category, hue level) rather than by position. Seaborn
    draws one container per hue level in `hue_order`, and the categories along the
    x-axis in tick order — so those two are what the lookup keys on, and the check
    turns a layout change into a failure rather than a silently misplaced error bar.

    Colour comes from the settings, following 02.2's grey-first rule: every hue
    level but the last is `settings.base_color`, the last — the one the plot is
    about — is `settings.highlight_color`. Pass `palette=` to override.
    """

    def build(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        hue: str,
        error: str,
        hue_order: list[str],
        **kwargs,
    ):
        kwargs.setdefault(
            "palette",
            [self.settings.base_color] * (len(hue_order) - 1)
            + [self.settings.highlight_color],
        )
        sns.barplot(
            data=data, x=x, y=y, hue=hue, hue_order=hue_order, ax=self.ax, **kwargs
        )
        if self.ax is None:
            raise ValueError("create_figure() must run before build()")

        intervals = data.set_index([x, hue])[error]
        categories = [label.get_text() for label in self.ax.get_xticklabels()]
        if len(self.ax.containers) != len(hue_order):
            raise ValueError(
                f"seaborn drew {len(self.ax.containers)} containers for "
                f"{len(hue_order)} hue levels; the error bars would be misplaced"
            )

        for container, level in zip(self.ax.containers, hue_order):
            for patch, category in zip(container, categories):
                self.ax.errorbar(
                    patch.get_x() + patch.get_width() / 2,
                    patch.get_height(),
                    yerr=intervals[(category, level)],
                    fmt="none",
                    ecolor="black",
                    capsize=4,
                )

        # Left to itself matplotlib puts this over the bars, which is the exact
        # thing 2.3 is about.
        self.ax.legend(loc="lower right", framealpha=1)
        return self.fig, self.ax


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

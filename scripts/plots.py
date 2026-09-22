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
import pandas as pd
import seaborn as sns
from goad_toolkit.visualizer import BasePlot
from matplotlib.lines import Line2D


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

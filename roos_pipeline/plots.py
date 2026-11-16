"""Roos's own plot classes for the six pipeline figures, as `BasePlot` subclasses.

`EventMarker` was written for roos_03.3 and moved here from `scripts/plots.py` (the
teacher's file) on 2026-10-06; the others were made for the pipeline. `HomeAwayPlot`
(les 7) and `StyleMapPlot` (les 6) came last, from roos_07.4 and roos_06.6. The plot
classes only the notebooks use moved to `scripts/roos_plots.py` on 2026-11-13.

Why subclass `BasePlot` instead of six lines of `plt.subplots` in a function:
`PlotSettings` carries the styling (figure size, labels, grey/highlight colours), so a
chart can be restyled without touching its drawing code, and `plot_on` stacks one
class on top of another (an `EventMarker` on a `LinePlot`) on the same axes.

Every class draws only. The numbers it draws (shares, fitted curves, p-values) are
computed in `aggregate.py` and passed in; colours come from a palette in
`settings.COLORS`, every text from `figures.py`.
"""

from __future__ import annotations

import textwrap

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from goad_toolkit.visualizer import BasePlot, LinePlot
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch

from roos_pipeline.settings import (
    COLORS,
    LIMITS,
    PATTERNS,
    HomePalette,
    Labelled,
    Side,
    StylePalette,
)
from scripts.plots import BarPlot


class EventMarker(BasePlot):
    """A quiet vertical line at one date, with an arrow and a label pointing at it.

    `VerticalDate` is a loud red dashed line that lands in the legend. This one is grey
    (so it recedes behind the data, per 02.2's grey-first rule) and carries its own
    label next to the line, so the legend does not need an entry for it.

    Drawn on top of an existing chart with `plot_on`:

        lines.plot_on(EventMarker(settings), date="2025-05-11", label="Benji is geboren")

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
    averages as `<topic>_rolling_avg` (what goad's `RollingAvg(rename=True)` writes);
    `labels` holds the legend text per topic.
    """

    def build(
        self,
        weekly: pd.DataFrame,
        trend: pd.DataFrame,
        x: str,
        colors: dict[str, str],
        labels: dict[str, str],
        events: dict[str, str],
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
                label=labels[topic],
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
        themes: dict[str, Labelled] = PATTERNS.home_themes,
        labelled: tuple[str, ...] = ("Parijs", "Nieuwegein"),
        min_share: float = LIMITS.home_min_share,
        palette: HomePalette = COLORS.home,
        max_width: float = 16.0,
        caption: str | None = None,
        caption_width: int = 175,
    ):
        if self.ax is None or self.fig is None:
            raise ValueError("create_figure() must run before build()")
        ax, fig = self.ax, self.fig
        fig.set_facecolor(palette.paper)
        ax.set_facecolor(palette.paper)
        color = palette.sides
        # the light outline that keeps a label readable on top of a line
        halo = [patheffects.withStroke(linewidth=4, foreground=palette.paper)]

        order = table.sort_values("thuis_score").index
        top = len(order) - 1
        place_pos = {
            p: (table.loc[p, "thuis_score"], top - i) for i, p in enumerate(order)
        }
        lo, hi = table["thuis_score"].min(), table["thuis_score"].max()
        spread = hi - lo
        left, right = lo - 0.55 * spread, hi + 0.55 * spread
        theme_pos = {}
        for group, x in ((Side.AWAY, left), (Side.HOME, right)):
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
                        color=palette.dark_grey,
                        path_effects=halo,
                        zorder=4,
                    )

        paper = {
            "boxstyle": "round,pad=0.25",
            "facecolor": palette.paper,
            "edgecolor": "none",
            "alpha": 0.9,
        }
        for place, (px, py) in place_pos.items():
            ax.scatter(
                px,
                py,
                s=160,
                color=palette.dark_grey,
                edgecolors=palette.paper,
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
                color=palette.grey,
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
                ha="right" if group == Side.AWAY else "left",
                va="center",
                fontsize=12.5,
                color="black",
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "facecolor": palette.paper,
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
                color=palette.grey,
            )

        ax.annotate(
            "",
            xy=(hi + 0.1 * spread, -0.75),
            xytext=(lo - 0.1 * spread, -0.75),
            arrowprops={"arrowstyle": "-|>", "color": palette.grey, "linewidth": 1.4},
        )
        ax.text(
            (lo + hi) / 2,
            -0.9,
            "hoe verder naar rechts, hoe meer de gesprekken gaan over thuis met kind",
            ha="center",
            va="top",
            fontsize=10.5,
            color=palette.grey,
        )

        ax.set_title(
            ""
        )  # BasePlot's centred title: this one sits on the left, above the subtitle
        ax.set_title(
            self.settings.title,
            loc="left",
            fontsize=18,
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
            color=palette.grey,
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
                color=palette.dark_grey,
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
    """Les 6: documents as small dots, half-year means as paths with arrows.

    `docs` has one row per document (`author`, `ts_local`, `x`, `y`), `means` one row
    per person per half-year. Colour is the person, lightness is time (light 2023,
    full 2026). `texts` holds every sentence on it (title, subtitle, the words at the
    ends of the axes and their examples, footnote), so the wording is decided in
    `figures.py`, not here. Moved from scripts/stijlkaart.py (`draw_stijlkaart`).
    """

    def build(
        self,
        docs: pd.DataFrame,
        means: pd.DataFrame,
        texts: dict[str, str],
        palette: StylePalette = COLORS.style,
        min_arrow_share: float = LIMITS.min_arrow_share,
    ):
        if self.ax is None or self.fig is None:
            raise ValueError("create_figure() must run before build()")
        fig, ax = self.fig, self.ax
        start, end = docs["ts_local"].min(), docs["ts_local"].max()

        def time_share(when: pd.Series) -> np.ndarray:
            return ((when - start) / (end - start)).to_numpy(dtype=float)

        fig.set_facecolor(palette.paper)
        ax.set_facecolor(palette.paper)
        ax.set_title("")
        ax.set_position((0.1, 0.2, 0.85, 0.62))  # room for the texts around it
        for spine in ax.spines.values():
            spine.set_color(palette.frame)
        ax.set_xticks([])
        ax.set_yticks([])

        for author, color in palette.persons.items():
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
        for author, color in palette.persons.items():
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
                    < min_arrow_share * x_range
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
                        "facecolor": palette.paper,
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
            color=palette.muted,
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
            color=palette.text,
        )
        ax.text(
            0,
            -0.085,
            texts["x_low_examples"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            color=palette.muted,
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
            color=palette.text,
        )
        ax.text(
            1,
            -0.085,
            texts["x_high_examples"],
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            color=palette.muted,
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
            color=palette.text,
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
            color=palette.muted,
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
            color=palette.text,
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
            color=palette.muted,
        )

        fig.text(
            0.1,
            0.955,
            texts["title"],
            ha="left",
            va="top",
            fontsize=18,
            fontweight="bold",
            color=palette.text,
        )
        fig.text(
            0.1,
            0.895,
            texts["subtitle"],
            ha="left",
            va="top",
            fontsize=13,
            color=palette.text,
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
            color=palette.muted,
            linespacing=1.6,
        )
        return fig, ax

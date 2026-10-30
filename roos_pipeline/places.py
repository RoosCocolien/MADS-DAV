"""Lesson 7 — which places come up together, and when?

Used by `notebooks/lesson7/roos_07.4-your-turn.ipynb`. The steps, in order:

    1_messages.parquet (one row per message)
      -> UnmergeMessages       a row the export glued together is split back into messages,
                               each with its own "[13-08-2023, 14:37:06]" time
      -> place_days()          grain changes: one row per (day, place) that came up
      -> place_graph()         one node per place, one edge per pair on the same day;
                               the weight is the number of days they shared
      -> prune()               the clutter rules: weak places, weak links, top-k links
      -> lane_layout()         x = the place's median day, y = a lane in its country's band
    -> TimelineGraphPlot

The edge rule is 07.1's "who was nearby" with a whole day as the window, on places
instead of people. Every count is in *days*, never in messages: ten messages planning one
train to Marseille are one day of Marseille.
"""

from __future__ import annotations

import re

import matplotlib.dates as mdates
import networkx as nx
import numpy as np
import pandas as pd
from goad_toolkit.datatransforms import TransformBase
from goad_toolkit.visualizer import BasePlot
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch

from roos_pipeline import settings

# "‎[13-08-2023, 14:37:06] Roos: " in the middle of a row: the start of another message.
EMBEDDED_HEADER = re.compile(
    r"‎?\[(\d{1,2}-\d{1,2}-\d{4}, \d{2}:\d{2}:\d{2})\] ([^:\n]+): "
)


class UnmergeMessages(TransformBase):
    """Split a row that holds several messages into one row per message.

    After an "afbeelding weggelaten" line the export sometimes runs the next messages into
    the same row, each behind its own "[dd-mm-yyyy, hh:mm:ss] Name: " header; 409 of those
    headers carry a different day than the row (counted 2026-10-30). Lesson 6 cut them off
    (`StripQuotedReply`); here they are kept, because a place in them was mentioned on the
    day in *their* header. If a header is a quote of an older message instead, its day was
    already counted then, and a day is only counted once per place, so nothing is doubled.
    """

    def transform(
        self,
        data: pd.DataFrame,
        column: str = "message",
        time_format: str = "%d-%m-%Y, %H:%M:%S",
    ) -> pd.DataFrame:
        rows = []
        for when, author, text in zip(
            data["ts_local"], data["author"], data[column].fillna(""), strict=True
        ):
            parts = EMBEDDED_HEADER.split(text)
            rows.append((when, author, parts[0]))
            for stamp, name, piece in zip(parts[1::3], parts[2::3], parts[3::3]):
                rows.append((pd.to_datetime(stamp, format=time_format), name, piece))
        out = pd.DataFrame(rows, columns=["ts_local", "author", column])
        out[column] = out[column].str.replace("‎", "", regex=False).str.strip()
        return out.sort_values("ts_local", kind="stable").reset_index(drop=True)


def place_days(
    messages: pd.DataFrame,
    places: dict[str, tuple[str, str]] = settings.PLACES,
    column: str = "message",
) -> pd.DataFrame:
    """One row per (day, place) that came up that day, whoever wrote it, however often."""
    day = messages["ts_local"].dt.normalize()
    text = messages[column].fillna("")
    found = [
        pd.DataFrame({"day": day[hit], "place": name, "country": country})
        for name, (country, pattern) in places.items()
        if (hit := text.str.contains(pattern, case=False, regex=True)).any()
    ]
    columns = ["day", "place", "country"]
    if not found:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(found)
        .drop_duplicates(["day", "place"])
        .sort_values(["day", "place"])
        .reset_index(drop=True)[columns]
    )


def place_graph(days: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> nx.Graph:
    """Places as nodes, an edge for every pair that came up on the same day in [start, end).

    Node attributes: `days` (how many days it came up), `country`, and `when` (the median
    of those days, which becomes its place on the time axis). Edge `weight`: how many days
    the two came up together.
    """
    window = days[(days["day"] >= start) & (days["day"] < end)]
    graph = nx.Graph()
    for place, rows in window.groupby("place"):
        graph.add_node(
            place,
            days=len(rows),
            country=rows["country"].iloc[0],
            when=rows["day"].sort_values().iloc[(len(rows) - 1) // 2],
        )
    for _, rows in window.groupby("day"):
        names = sorted(rows["place"])
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                if graph.has_edge(first, second):
                    graph[first][second]["weight"] += 1
                else:
                    graph.add_edge(first, second, weight=1)
    return graph


def prune(
    graph: nx.Graph,
    min_days: int = settings.MIN_PLACE_DAYS,
    min_link: int = settings.MIN_LINK_DAYS,
    max_links: int = settings.MAX_LINKS,
    keep: tuple[str, ...] = settings.ALWAYS_KEEP,
    both_ends: bool = False,
    no_loners: bool = True,
) -> nx.Graph:
    """The clutter rules, in this order: places, weak links, the top-k per place, loners.

    1. A place stays if it came up on at least `min_days` days, or is in `keep`.
    2. A link stays if the two came up together on at least `min_link` days.
    3. Each place keeps its `max_links` strongest links. By default a link survives if it
       is among the strongest of *one* of its places: a small place keeps its link to
       Utrecht, and Utrecht can end up with more than `max_links`. With `both_ends` it
       must be among the strongest of both, so no place has more than `max_links` (too
       strict on the real chat, 2026-10-30: 6 links, 11 places left alone).
    4. With `no_loners`, a place left without any link gets back its single strongest
       link from after step 1, even below `min_link` (Corsica: 4 days with Paris).

    A tie goes to the neighbour that came up on more days, then alphabetically, so the
    result is fixed. The order matters: step 3 ranks only the links left after 1 and 2,
    so a big place's slot is never taken by a link to a place that was dropped anyway.
    """
    places = [
        n for n, d in graph.nodes(data=True) if d["days"] >= min_days or n in keep
    ]
    kept = nx.Graph(graph.subgraph(places))
    small = kept.copy()
    small.remove_edges_from(
        [(u, v) for u, v, w in small.edges(data="weight") if w < min_link]
    )

    def ranked(g: nx.Graph, node: str) -> list[str]:
        return sorted(
            g[node],
            key=lambda other: (
                -g[node][other]["weight"],
                -g.nodes[other]["days"],
                other,
            ),
        )

    top = {node: set(ranked(small, node)[:max_links]) for node in small.nodes()}

    def drop(u: str, v: str) -> bool:
        if both_ends:
            return v not in top[u] or u not in top[v]
        return v not in top[u] and u not in top[v]

    small.remove_edges_from([(u, v) for u, v in small.edges() if drop(u, v)])
    if no_loners:
        for node in [n for n in small if small.degree(n) == 0 and kept.degree(n) > 0]:
            other = ranked(kept, node)[0]
            small.add_edge(node, other, weight=kept[node][other]["weight"])
    return small


def lane_layout(
    graph: nx.Graph,
    countries: list[str] = settings.COUNTRY_ORDER,
    min_gap: pd.Timedelta = pd.Timedelta(days=45),
    lane_height: float = 0.16,
    big_days: int = 60,
) -> dict[str, tuple[float, float]]:
    """x = the median day a place came up; y = a lane inside its country's band.

    Bands sit at y = 2, 1, 0 for France, the Netherlands and elsewhere. Within a band, the
    biggest place goes in the middle lane and every next one in the lane nearest the
    middle where no place lies within `min_gap` days of it, so dots and labels don't sit
    on top of each other. A place of `big_days` or more has a dot wider than a lane, so it
    also claims the lanes on either side. **y means nothing beyond the country**; x is data.
    """
    centre = {country: len(countries) - 1 - i for i, country in enumerate(countries)}
    offsets = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
    taken: dict[tuple[str, int], list[pd.Timestamp]] = {}
    pos = {}
    by_size = sorted(graph.nodes(data=True), key=lambda nd: (-nd[1]["days"], nd[0]))
    for node, data in by_size:
        country, when = data["country"], data["when"]
        for lane in offsets:
            if all(
                abs(when - other) >= min_gap for other in taken.get((country, lane), [])
            ):
                break
        spread = (-1, 0, 1) if data["days"] >= big_days else (0,)
        for step in spread:
            taken.setdefault((country, lane + step), []).append(when)
        pos[node] = (
            mdates.date2num(data["when"]),
            centre[data["country"]] + lane * lane_height,
        )
    return pos


# --- plots -------------------------------------------------------------------------


class TimelineGraphPlot(BasePlot):
    """A network laid out on a time axis: dots are places, lines are shared days.

    A dot's area is proportional to the days its place came up, its colour is the
    country, and its x is the median of those days. A line's width and darkness grow with
    the days the two places came up together; the heaviest link gets `max_width` points.
    `shade` greys a period (start, end, label), and `label_min` hides the names of places
    below that many days, so a crowded graph stays readable; every dot is still drawn.

    The y-axis has no labels: the colour already says the country, and the bands only
    keep the countries apart (Roos, 2026-10-30). The legend shows only the colours.
    """

    def build(
        self,
        graph: nx.Graph,
        pos: dict[str, tuple[float, float]],
        shade: tuple[pd.Timestamp, pd.Timestamp, str] | None = None,
        label_min: int = 0,
        size_per_day: float = 6.0,
        max_width: float = 8.0,
        colors: dict[str, str] = settings.COUNTRY_COLORS,
        countries: list[str] = settings.COUNTRY_ORDER,
        xlim: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ):
        if self.ax is None:
            raise ValueError("create_figure() must run before build()")
        ax = self.ax

        if shade is not None:
            # matplotlib's x-axis counts days as numbers, like the positions in `pos`
            start, end = mdates.date2num(shade[0]), mdates.date2num(shade[1])
            ax.axvspan(
                start, end, color=settings.LIGHT_GREY, alpha=0.45, lw=0, zorder=0
            )
            # the label sits in the middle of the part of the period that is on screen
            if xlim is not None:
                start = max(start, mdates.date2num(xlim[0]))
                end = min(end, mdates.date2num(xlim[1]))
            ax.text(
                start + (end - start) / 2,
                len(countries) - 0.35,
                shade[2],
                ha="center",
                va="bottom",
                color=settings.DARK_GREY,
                fontsize=10,
                style="italic",
            )

        heaviest = max((w for _, _, w in graph.edges(data="weight")), default=1)
        for u, v, w in sorted(graph.edges(data="weight"), key=lambda e: e[2]):
            ax.add_patch(
                FancyArrowPatch(
                    pos[u],
                    pos[v],
                    arrowstyle="-",
                    connectionstyle="arc3,rad=0.15",
                    linewidth=0.4 + (max_width - 0.4) * w / heaviest,
                    color=settings.GREY,
                    alpha=0.15 + 0.5 * w / heaviest,
                    zorder=1,
                )
            )

        nodes = list(graph.nodes(data=True))
        ax.scatter(
            [pos[n][0] for n, _ in nodes],
            [pos[n][1] for n, _ in nodes],
            s=[d["days"] * size_per_day for _, d in nodes],
            c=[colors[d["country"]] for _, d in nodes],
            edgecolors="white",
            linewidths=0.8,
            alpha=0.9,
            zorder=2,
        )
        for node, data in nodes:
            if data["days"] >= label_min:
                x, y = pos[node]
                radius = np.sqrt(data["days"] * size_per_day) / 2
                ax.annotate(
                    node,
                    (x, y),
                    xytext=(radius + 2, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=settings.DARK_GREY,
                    zorder=3,
                )

        # a label every two months, "Sep 23", "Nov 23", ...; a small tick for every month
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        if xlim is not None:
            ax.set_xlim(mdates.date2num(xlim[0]), mdates.date2num(xlim[1]))
        ax.set_yticks([])
        ax.set_ylim(-0.75, len(countries) - 0.1)
        ax.spines[["top", "right", "left"]].set_visible(False)

        handles: list[Artist] = [
            Line2D([], [], marker="o", ls="", color=colors[c], markersize=8, label=c)
            for c in countries
        ]
        if shade is not None:
            handles.append(Patch(color=settings.LIGHT_GREY, alpha=0.45, label=shade[2]))
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            frameon=False,
            labelspacing=1.2,
        )
        return self.fig, ax

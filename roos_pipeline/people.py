"""Lesson 7 — who is named near Benji? Looking for babysitters.

Used by `notebooks/lesson7/roos_07.4-your-turn.ipynb` (§7.4.10). The steps, in order:

    messages (after UnmergeMessages + CanonicalAuthor)
      -> mentions()        one row per (message, person) named in it, with its time
      -> benji_graph()     Benji in the middle, one node per person; the edge weight is the
                           number of days the person was named within a window of "Benji"
      -> star_layout()     Benji in the centre, the others on a circle, strongest at the top
    -> places.GraphPlot

Who connects our friends and family (§7.4.11) reuses `mentions()`:

      -> people_graph()    an edge for two people named in the same message, or within a
                           window of each other; the weight is the days that happened
      -> person_sides()    Roos's side, Louis's side or shared, by who names them (60/40)
    -> nx.betweenness_centrality, places.circle_layout(attr="side"), places.GraphPlot
    -> side_order()       the same people, grouped by side, for MatrixPlot

The people come from config.toml `[people]`, not from settings: they are names of family
and friends. `[people.names]` counts whoever writes it; `[people.Roos]` and
`[people.Louis]` only count when that person writes it, because "mon père" from Roos is
someone else than "mon père" from Louis.
"""

from __future__ import annotations

import textwrap
import tomllib
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from goad_toolkit.visualizer import BasePlot
from matplotlib.colors import LogNorm

from roos_pipeline import settings
from roos_pipeline.places import who_names

NAME, FAMILY = "name", "family"


def load_people(config: Path) -> dict[str, dict[str, str]]:
    """Read `[people]` from config.toml: {"names" | author: {person: regex}}."""
    with config.open("rb") as f:
        return tomllib.load(f).get("people", {})


def mentions(
    messages: pd.DataFrame,
    people: dict[str, dict[str, str]],
    column: str = "message",
) -> pd.DataFrame:
    """One row per (message, person) named in it.

    Columns: `message_id` (the row label in `messages`), `ts_local`, `author`, `person`,
    `kind`. `kind` is "name" for `people["names"]`, which count whoever writes them, and
    "family" for the tables under an author's name, which only count in that author's
    messages. A person named twice in one message is one row.
    """
    text = messages[column].fillna("")
    found = []
    for table, patterns in people.items():
        mine = (
            pd.Series(True, index=messages.index)
            if table == "names"
            else messages["author"] == table
        )
        kind = NAME if table == "names" else FAMILY
        for person, pattern in patterns.items():
            hit = mine & text.str.contains(pattern, case=False, regex=True)
            if hit.any():
                found.append(
                    pd.DataFrame(
                        {
                            "ts_local": messages.loc[hit, "ts_local"],
                            "author": messages.loc[hit, "author"],
                            "person": person,
                            "kind": kind,
                        }
                    )
                )
    columns = ["message_id", "ts_local", "author", "person", "kind"]
    if not found:
        return pd.DataFrame(columns=columns)
    out = pd.concat(found).reset_index(names="message_id")
    return (
        out.drop_duplicates(["message_id", "person"])
        .sort_values(["ts_local", "person"])
        .reset_index(drop=True)[columns]
    )


def near(
    times: pd.Series, anchors: pd.Series, window: pd.Timedelta = settings.BENJI_WINDOW
) -> pd.Series:
    """True where a time lies within `window` before or after any of `anchors`."""
    sorted_anchors = np.sort(anchors.to_numpy())
    if len(sorted_anchors) == 0:
        return pd.Series(False, index=times.index)
    first_after = np.searchsorted(sorted_anchors, (times - window).to_numpy())
    candidate = sorted_anchors[np.minimum(first_after, len(sorted_anchors) - 1)]
    found = (first_after < len(sorted_anchors)) & (
        candidate <= (times + window).to_numpy()
    )
    return pd.Series(found, index=times.index)


def benji_graph(
    named: pd.DataFrame,
    benji_times: pd.Series,
    window: pd.Timedelta = settings.BENJI_WINDOW,
    min_near: int = settings.MIN_NEAR_DAYS,
    centre: str = settings.BENJI,
) -> nx.Graph:
    """A star: `centre` in the middle, an edge to every person named near it.

    `named` is `mentions()` cut to the period you want, `benji_times` the times of the
    messages naming Benji in that period. Node `days`: the days a person was named at
    all (Benji: the days Benji was named); edge `weight`: the days on which the person
    was named within `window` of a Benji message. A person needs `min_near` such days.
    Node `country` holds the kind ("name", "family", "centre"), so `GraphPlot` and the
    colouring can treat a family word differently from a name.
    """
    days = named.assign(day=named["ts_local"].dt.normalize())
    close = days[near(days["ts_local"], benji_times, window)]
    per_person = pd.DataFrame(
        {
            "days": days.groupby(["person", "kind"])["day"].nunique(),
            "near_days": close.groupby(["person", "kind"])["day"].nunique(),
        }
    ).fillna(0)
    graph = nx.Graph()
    graph.add_node(centre, days=benji_times.dt.normalize().nunique(), country="centre")
    for row in per_person.reset_index().to_dict("records"):
        if row["near_days"] >= min_near:
            graph.add_node(row["person"], days=int(row["days"]), country=row["kind"])
            graph.add_edge(centre, row["person"], weight=int(row["near_days"]))
    return graph


def star_layout(
    graph: nx.Graph, centre: str = settings.BENJI
) -> dict[str, tuple[float, float]]:
    """`centre` at (0, 0); the rest on a circle, clockwise from the top, strongest first.

    The position says the rank of the link to the centre, nothing else. One slot at the
    top stays empty, so the strongest and the weakest don't end up side by side.
    """
    others = sorted(
        (n for n in graph if n != centre),
        key=lambda n: (-graph[centre][n]["weight"], n),
    )
    step = 2 * np.pi / (len(others) + 1)
    pos = {centre: (0.0, 0.0)}
    for i, node in enumerate(others):
        angle = np.pi / 2 - (i + 0.5) * step
        pos[node] = (float(np.cos(angle)), float(np.sin(angle)))
    return pos


def person_sides(named: pd.DataFrame) -> pd.Series:
    """Per person: "Roos", "Louis" or "shared", by `places.who_names` on the days.

    Roos's share = the days Roos named them / (Roos's days + Louis's days), the same rule
    as for places: a day on which both named someone counts for both.
    """
    days = named.assign(day=named["ts_local"].dt.normalize())
    per_author = (
        days.drop_duplicates(["person", "author", "day"])
        .groupby(["person", "author"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Roos", "Louis"], fill_value=0)
    )
    share = per_author["Roos"] / per_author.sum(axis=1)
    return share.map(who_names).rename("side")


def people_graph(named: pd.DataFrame, window: pd.Timedelta | None = None) -> nx.Graph:
    """People as nodes, an edge for two people named close together.

    With `window=None` "close" means in the same message; with a window, within that
    time of each other, whoever wrote it (07.1's "who was nearby", on names instead of
    authors). Node `days`: the days a person was named; `kind`: name or family word.
    Edge `weight`: the days on which the two were close; a pair across midnight counts
    on the day of the later mention.
    """
    days = named.assign(day=named["ts_local"].dt.normalize())
    graph = nx.Graph()
    for (person, kind), rows in days.groupby(["person", "kind"]):
        graph.add_node(person, days=rows["day"].nunique(), kind=kind)

    pairs: set[tuple[str, str, pd.Timestamp]] = set()
    if window is None:
        for _, rows in days.groupby("message_id"):
            names = sorted(set(rows["person"]))
            day = rows["day"].iloc[0]  # one message, one day
            for i, first in enumerate(names):
                pairs.update((first, second, day) for second in names[i + 1 :])
    else:
        ordered = days.sort_values("ts_local")
        times = ordered["ts_local"].to_list()
        people = ordered["person"].to_list()
        day_of = ordered["day"].to_list()
        left = 0
        for right in range(len(ordered)):
            while times[right] - times[left] > window:
                left += 1
            for i in range(left, right):
                if people[i] != people[right]:
                    first, second = sorted((people[i], people[right]))
                    pairs.add((first, second, day_of[right]))

    for first, second, _ in pairs:
        if graph.has_edge(first, second):
            graph[first][second]["weight"] += 1
        else:
            graph.add_edge(first, second, weight=1)
    return graph


def side_order(
    graph: nx.Graph,
    sides: pd.Series,
    order: tuple[str, ...] = ("Roos", "shared", "Louis"),
) -> list[str]:
    """The nodes grouped by side in `order`, the most-named first within a side."""
    return sorted(
        graph,
        key=lambda n: (order.index(sides[n]), -graph.nodes[n]["days"], n),
    )


class MatrixPlot(BasePlot):
    """The same graph as an adjacency matrix: a row and a column per person.

    A cell's colour is the days the two were named close together (white: never), on a
    log scale, because a few pairs have 20+ days and most have 1 or 2. `order` sets the
    rows; thin lines separate the sides, so a block off the diagonal is "Roos's people
    with Louis's people". Names are coloured by `label_colors`, the `marked` names bold.
    Unlike the network, the matrix needs no pruning: an empty cell costs no ink.
    """

    def build(
        self,
        graph: nx.Graph,
        order: list[str],
        label_colors: dict[str, str],
        groups: list[int] | None = None,
        marked: tuple[str, ...] = (),
        caption: str | None = None,
        cmap: str = "Blues",
        caption_width: int = 110,
    ):
        if self.ax is None or self.fig is None:
            raise ValueError("create_figure() must run before build()")
        ax = self.ax
        matrix = nx.to_numpy_array(graph, nodelist=order, weight="weight")
        shown = np.ma.masked_equal(matrix, 0)
        image = ax.imshow(
            shown,
            cmap=cmap,
            norm=LogNorm(vmin=1, vmax=max(matrix.max(), 2)),
            interpolation="nearest",
        )
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
        ax.set_yticks(range(len(order)), order, fontsize=7)
        for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
            for label in labels:
                label.set_color(label_colors[label.get_text()])
                if label.get_text() in marked:
                    label.set_fontweight("bold")
        for edge in np.cumsum(groups or [])[:-1]:
            ax.axhline(edge - 0.5, color=settings.DARK_GREY, linewidth=0.8)
            ax.axvline(edge - 0.5, color=settings.DARK_GREY, linewidth=0.8)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        bar = self.fig.colorbar(image, ax=ax, shrink=0.6, pad=0.02)
        bar.set_label("dagen samen genoemd (log-schaal)")
        if caption is not None:
            ax.text(
                0.5,
                -0.16,
                textwrap.fill(caption, caption_width),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                color=settings.DARK_GREY,
            )
        return self.fig, ax

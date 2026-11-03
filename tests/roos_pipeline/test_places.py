"""Unit tests for the lesson 7 steps (`roos_pipeline/places.py`), on hand-made rows."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from roos_pipeline.places import (
    UnmergeMessages,
    author_shares,
    circle_layout,
    lane_layout,
    place_days,
    place_graph,
    prune,
    who_names,
)

PLACES = {
    "Paris": ("France", r"\bparis\b"),
    "Corsica": ("France", r"\bcorse\b|\bcorsica\b"),
    "Utrecht": ("Netherlands", r"\butrecht?\b"),
    "Basel": ("Elsewhere", r"\bbasel\b"),
}


def messages(*rows: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_local": pd.to_datetime([when for when, _ in rows]),
            "author": "Roos",
            "message": [text for _, text in rows],
        }
    )


def test_glued_messages_are_split_and_get_their_own_day():
    data = messages(
        (
            "2023-08-13 14:33",
            "Un velo? ‎[14-08-2023, 09:00:00] Louis: Toujours en Corse",
        ),
    )
    out = UnmergeMessages()(data)
    assert out["message"].tolist() == ["Un velo?", "Toujours en Corse"]
    assert out["author"].tolist() == ["Roos", "Louis"]
    assert out["ts_local"].dt.day.tolist() == [13, 14]


def test_a_place_counts_once_per_day_and_corse_is_not_corse_accent():
    data = messages(
        ("2024-01-01 09:00", "Paris paris PARIS"),
        ("2024-01-01 21:00", "back in Paris"),
        ("2024-01-02 10:00", "un café corsé"),
        ("2024-01-02 11:00", "Utrech!"),
    )
    days = place_days(data, places=PLACES)
    assert days[["place"]].value_counts().to_dict() == {("Paris",): 1, ("Utrecht",): 1}


def test_edge_weight_is_days_together_and_end_is_excluded():
    data = messages(
        ("2024-01-01 09:00", "Paris and Utrecht"),
        ("2024-01-02 09:00", "Paris"),
        ("2024-01-02 22:00", "Utrecht, then Basel"),
        ("2024-01-03 09:00", "Paris Utrecht"),  # on the end date: left out
    )
    graph = place_graph(
        place_days(data, places=PLACES),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-03"),
    )
    assert graph["Paris"]["Utrecht"]["weight"] == 2
    assert graph["Utrecht"]["Basel"]["weight"] == 1
    assert graph.nodes["Paris"]["days"] == 2
    assert graph.nodes["Paris"]["when"] == pd.Timestamp("2024-01-01")  # lower median


def star() -> nx.Graph:
    """A hub with four spokes of weight 9, 8, 7, 6, one weak spoke, and a quiet Corsica."""
    graph = nx.Graph()
    for name, days in [
        ("hub", 50),
        ("a", 20),
        ("b", 20),
        ("c", 20),
        ("d", 20),
        ("weak", 20),
        ("tiny", 3),
        ("Corsica", 2),
    ]:
        graph.add_node(
            name, days=days, country="France", when=pd.Timestamp("2024-01-01")
        )
    for name, w in [("a", 9), ("b", 8), ("c", 7), ("d", 6), ("weak", 4), ("tiny", 9)]:
        graph.add_edge("hub", name, weight=w)
    return graph


def test_prune_strict_keeps_three_strongest_on_both_ends_and_corsica():
    small = prune(
        star(),
        min_days=10,
        min_link=5,
        max_links=3,
        keep=("Corsica",),
        both_ends=True,
        no_loners=False,
    )
    assert set(small.nodes()) == {"hub", "a", "b", "c", "d", "weak", "Corsica"}
    assert sorted(small["hub"]) == ["a", "b", "c"]
    assert small.degree("d") == 0  # its only link is the hub's fourth


def test_prune_either_end_lets_the_hub_keep_more():
    small = prune(
        star(), min_days=10, min_link=5, max_links=3, keep=(), no_loners=False
    )
    assert sorted(small["hub"]) == ["a", "b", "c", "d"]


def test_a_loner_gets_its_strongest_link_back_even_below_the_threshold():
    graph = star()
    graph.add_edge("Corsica", "a", weight=2)
    graph.add_edge("Corsica", "hub", weight=2)  # a tie: the hub came up on more days
    small = prune(graph, min_days=10, min_link=5, max_links=3, keep=("Corsica",))
    assert sorted(small["Corsica"]) == ["hub"]
    assert small["hub"]["weak"]["weight"] == 4  # below min_link, but weak was alone
    assert nx.is_connected(small)


def test_lanes_keep_close_places_apart():
    graph = star()
    pos = lane_layout(graph, countries=["France"])
    ys = [y for _, y in pos.values()]
    assert (
        len(set(ys)) == graph.number_of_nodes()
    )  # same day, so every one its own lane
    assert pos["hub"][1] == 0  # the biggest sits in the middle


def test_a_big_place_also_claims_the_lanes_next_to_it():
    graph = star()
    graph.nodes["hub"]["days"] = 200
    pos = lane_layout(graph, countries=["France"], lane_height=1.0, big_days=60)
    assert sorted(abs(y) for n, (_, y) in pos.items() if n != "hub")[0] == 2


def test_author_share_counts_days_per_person_and_a_shared_day_for_both():
    data = pd.DataFrame(
        {
            "ts_local": pd.to_datetime(
                [
                    "2024-01-01 09:00",
                    "2024-01-01 10:00",
                    "2024-01-01 11:00",
                    "2024-01-02 09:00",
                ]
            ),
            "author": ["Roos", "Roos", "Louis", "Roos"],
            "message": ["Paris", "Paris again", "Paris", "Paris"],
        }
    )
    shares = author_shares(data, places=PLACES)
    assert (shares.loc["Paris", "Roos"], shares.loc["Paris", "Louis"]) == (2, 1)
    assert shares.loc["Paris", "share"] == 2 / 3


def test_who_names_includes_both_boundaries_on_the_owners_side():
    assert who_names(0.6) == "Roos"
    assert who_names(0.4) == "Louis"
    assert [who_names(s) for s in (0.41, 0.5, 0.59)] == ["shared"] * 3


def test_circle_puts_the_two_biggest_hubs_at_the_top_and_countries_on_their_side():
    graph = nx.Graph()
    for name, country, days in [
        ("Paris", "France", 50),
        ("Corsica", "France", 5),
        ("Utrecht", "Netherlands", 60),
        ("Gouda", "Netherlands", 7),
        ("Basel", "Elsewhere", 9),
    ]:
        graph.add_node(name, country=country, days=days)
    pos = circle_layout(graph)
    assert pos["Utrecht"][0] > 0 > pos["Paris"][0]  # right and left of the top
    assert min(pos["Utrecht"][1], pos["Paris"][1]) > max(
        pos[n][1] for n in ("Corsica", "Gouda", "Basel")
    )
    assert pos["Basel"][1] < 0  # elsewhere at the bottom

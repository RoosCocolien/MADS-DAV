"""Unit tests for people near Benji (`scripts/roos_people.py`), on hand-made rows."""

from __future__ import annotations

import pandas as pd

from scripts.roos_people import (
    benji_graph,
    mentions,
    near,
    people_graph,
    person_sides,
    star_layout,
)

PEOPLE = {
    "names": {"Janneke": r"\bjanneke\b", "Ton": r"(?-i:\bTon\b)"},
    "Roos": {"vader Roos": r"\bmon p[eè]re\b"},
    "Louis": {"vader Roos": r"\bton p[eè]re\b", "vader Louis": r"\bmon p[eè]re\b"},
}


def messages(*rows: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_local": pd.to_datetime([when for when, _, _ in rows]),
            "author": [author for _, author, _ in rows],
            "message": [text for _, _, text in rows],
        }
    )


def test_a_family_word_means_someone_else_depending_on_who_writes_it():
    data = messages(
        ("2025-06-01 09:00", "Roos", "mon père vient"),
        ("2025-06-01 09:05", "Louis", "mon père aussi, et ton père"),
    )
    found = mentions(data, PEOPLE)
    assert sorted(found["person"]) == ["vader Louis", "vader Roos", "vader Roos"]
    assert set(found["kind"]) == {"family"}


def test_a_case_sensitive_name_skips_the_ordinary_word():
    data = messages(
        ("2025-06-01 09:00", "Louis", "ton père et Ton"),
        ("2025-06-01 09:01", "Roos", "Janneke JANNEKE"),
    )
    found = mentions(data, {"names": PEOPLE["names"]})
    assert found["person"].tolist() == ["Ton", "Janneke"]  # Janneke once per message


def test_near_includes_both_edges_of_the_window():
    anchors = pd.Series(pd.to_datetime(["2025-06-01 12:00"]))
    times = pd.Series(
        pd.to_datetime(["2025-06-01 11:30", "2025-06-01 12:30", "2025-06-01 12:31"])
    )
    window = pd.Timedelta(minutes=30)
    assert near(times, anchors, window).tolist() == [True, True, False]


def test_edge_weight_counts_days_near_benji_and_a_threshold_drops_the_rest():
    data = messages(
        ("2025-06-01 09:00", "Roos", "Benji slaapt"),
        ("2025-06-01 09:10", "Roos", "Janneke kan zaterdag"),
        ("2025-06-01 09:20", "Roos", "Janneke echt"),  # same day: still one day
        ("2025-06-02 09:00", "Louis", "Benji"),
        ("2025-06-02 09:20", "Roos", "Janneke ok"),
        ("2025-06-03 18:00", "Roos", "Janneke, nobody else"),
        ("2025-06-03 08:00", "Roos", "mon père"),  # not near any Benji message
    )
    named = mentions(data, PEOPLE)
    benji = data.loc[data["message"].str.contains("Benji"), "ts_local"]
    graph = benji_graph(named, benji, pd.Timedelta(minutes=30), min_near=1)
    assert graph["Benji"]["Janneke"]["weight"] == 2
    assert graph.nodes["Janneke"]["days"] == 3
    assert "vader Roos" not in graph  # 0 days near Benji
    assert graph.nodes["Benji"]["days"] == 2


def test_the_strongest_link_sits_at_the_top():
    data = messages(
        ("2025-06-01 09:00", "Roos", "Benji Janneke"),
        ("2025-06-02 09:00", "Roos", "Benji Janneke Ton"),
    )
    named = mentions(data, PEOPLE)
    benji = data["ts_local"]
    graph = benji_graph(named, benji, pd.Timedelta(minutes=30), min_near=1)
    pos = star_layout(graph)
    assert pos["Benji"] == (0.0, 0.0)
    assert pos["Janneke"][1] > pos["Ton"][1]


def test_people_are_linked_in_the_same_message_or_within_the_window():
    data = messages(
        ("2025-06-01 09:00", "Roos", "Janneke en Ton"),
        ("2025-06-01 09:20", "Louis", "mon père"),
        ("2025-06-02 09:00", "Roos", "Janneke en Ton"),
    )
    named = mentions(data, PEOPLE)
    same_message = people_graph(named)
    assert same_message["Janneke"]["Ton"]["weight"] == 2
    assert not same_message.has_edge("Janneke", "vader Louis")
    nearby = people_graph(named, pd.Timedelta(minutes=30))
    assert nearby["Janneke"]["vader Louis"]["weight"] == 1
    assert nearby.nodes["Janneke"]["days"] == 2


def test_a_side_is_who_names_someone_on_more_days():
    data = messages(
        ("2025-06-01 09:00", "Roos", "Janneke"),
        ("2025-06-02 09:00", "Roos", "Janneke"),
        ("2025-06-03 09:00", "Louis", "Janneke en Ton"),
        ("2025-06-04 09:00", "Roos", "Ton"),
    )
    sides = person_sides(mentions(data, PEOPLE))
    assert sides.to_dict() == {"Janneke": "Roos", "Ton": "shared"}  # 2/3, 1/2

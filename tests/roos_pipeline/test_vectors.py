"""Unit tests for the lesson 6 steps (`roos_pipeline/vectors.py`), on hand-made rows.

No embedding model here: the steps around it (quotes, units, the shuffle test) are where
a wrong number can hide, and they run in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from roos_pipeline.vectors import (
    StripQuotedReply,
    WeekStart,
    WordCount,
    cluster_contrast,
    cluster_keywords,
    contrast_intervals,
    language_units,
    neighbour_agreement,
    topic_labels,
    weekly_cluster_shares,
)


def test_quoted_reply_is_cut_off():
    data = pd.DataFrame(
        {
            "message": [
                "Too expensive ‎[28-08-2024, 18:21:37] Roos: Groetjes",
                "Oui",
                None,
            ]
        }
    )
    out = StripQuotedReply()(data)
    assert out["message"].tolist() == ["Too expensive", "Oui", ""]


def test_week_starts_on_monday():
    data = pd.DataFrame(
        {"ts_local": pd.to_datetime(["2024-08-25 23:00", "2024-08-26 01:00"])}
    )
    out = WeekStart()(data)
    # Sunday the 25th belongs to the week of Monday the 19th; Monday the 26th starts its own.
    assert out["week"].tolist() == [
        pd.Timestamp("2024-08-19"),
        pd.Timestamp("2024-08-26"),
    ]


def test_word_count_bands():
    data = pd.DataFrame(
        {"message": ["un deux trois quatre", " ".join(["w"] * 9), " ".join(["w"] * 41)]}
    )
    out = WordCount(bands=(8, 20, 40))(data)
    assert out["n_words"].tolist() == [4, 9, 41]
    assert out["length_band"].tolist() == ["1-8", "9-20", "41+"]


def test_language_switch_stays_one_session_but_two_units():
    start = pd.Timestamp("2024-08-26 12:00")
    data = pd.DataFrame(
        {
            "ts_local": [start + pd.Timedelta(minutes=m) for m in (0, 1, 2, 3, 120)],
            "author": "Louis",
            "message": ["see you", "je t'aime", "à ce soir", "dag", "home now"],
            "language": ["en", "fr", "fr", "nl", "en"],
        }
    )
    data = WeekStart()(data)
    units = language_units(data, threshold=600)
    # Minutes 0-3: one session, split into an English and a French unit (Dutch dropped).
    # Minute 120: a new session after a two-hour gap.
    assert units[["session_id", "language", "n_messages"]].values.tolist() == [
        [0, "en", 1],
        [0, "fr", 2],
        [1, "en", 1],
    ]
    assert units.loc[1, "message"] == "je t'aime à ce soir"


def contrast_units(
    french_cluster: list[int], english_cluster: list[int]
) -> pd.DataFrame:
    rows = [{"language": "fr", "cluster": c} for c in french_cluster]
    rows += [{"language": "en", "cluster": c} for c in english_cluster]
    data = pd.DataFrame(rows)
    data["week"] = pd.Timestamp("2024-08-26")
    data["length_band"] = "9-20"
    return data


def test_contrast_finds_a_real_difference():
    units = contrast_units([0] * 18 + [1] * 2, [0] * 2 + [1] * 18)
    out = cluster_contrast(units, n_shuffle=2000, seed=0)
    assert out["difference"].iloc[0] == 0.8  # 90% of French vs 10% of English
    assert out["p"].iloc[0] < 0.001


def test_contrast_explains_away_a_length_difference():
    # Within each band French and English look the same; French is just shorter.
    short = contrast_units([0] * 9 + [1], [0] * 9 + [1]).assign(length_band="1-8")
    long_ = contrast_units([1] * 9 + [0], [1] * 9 + [0]).assign(length_band="41+")
    units = pd.concat([short, short, short, long_]).reset_index(drop=True)
    units = pd.concat([units, long_.assign(language="en")]).reset_index(drop=True)
    out = cluster_contrast(units, n_shuffle=2000, seed=0)
    assert out["difference"].iloc[0] > 0.1  # raw: French looks more "cluster 0"
    assert np.isclose(out["beyond_strata"].iloc[0], 0, atol=0.03)  # but that is length
    assert out["p"].iloc[0] > 0.3


def test_keywords_skip_words_every_cluster_uses():
    texts = pd.Series(
        ["the train leaves at nine"] * 5
        + ["the love of my life"] * 5
        + ["the end"] * 90
    )
    labels = np.array([0] * 5 + [1] * 5 + [2] * 90)
    out = cluster_keywords(texts, labels, n=2, max_share=0.06)
    assert "the" not in out[0] + out[1]  # in every unit: no help telling clusters apart
    assert set(out[0]) <= {"train", "leaves", "nine"}
    assert set(out[1]) == {"love", "life"}


def test_intervals_hold_a_difference_that_every_week_shows():
    weeks = [contrast_units([0] * 9 + [1], [0] + [1] * 9) for _ in range(6)]
    for i, week in enumerate(weeks):
        week["week"] = pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=5 * i)
    out = contrast_intervals(pd.concat(weeks), n_boot=200, seed=0)
    # 90% of French vs 10% of English in every week, so every resample gives +80 points.
    assert np.allclose(
        out.loc[0, ["difference", "low", "high", "odd_months", "even_months"]], 80
    )


def test_weekly_shares_skip_thin_weeks_and_keep_zeros():
    units = contrast_units([0, 0, 0, 1, 1], [0])
    out = weekly_cluster_shares(units, min_units=5)
    # English has 1 unit that week: left out. French: 3/5 in cluster 0, 2/5 in cluster 1.
    assert set(out["language"]) == {"fr"}
    assert out.set_index("cluster")["share"].to_dict() == {0: 0.6, 1: 0.4}


def test_topic_labels_ignore_accents_and_mark_both():
    texts = pd.Series(
        [
            "Je pense très fort à toi",
            "My train leaves at 18:30",
            "Good luck, the train is late",
            "Hahaha ok",
            None,
        ]
    )
    out = topic_labels(texts)
    assert out.tolist() == ["love", "planning", "both", "other", "other"]


def test_topic_labels_skip_french_car():
    # French "car" means "because", not a car.
    assert topic_labels(
        pd.Series(["c'est bien car tu es là", "I took the car"])
    ).tolist() == [
        "other",
        "planning",
    ]


def test_neighbours_agree_when_topics_sit_apart():
    rng = np.random.default_rng(0)
    vectors = np.vstack(
        [rng.normal([5, 0], 0.1, (10, 2)), rng.normal([0, 5], 0.1, (10, 2))]
    )
    out = neighbour_agreement(vectors, np.array(["love"] * 10 + ["planning"] * 10), k=5)
    assert out["neighbours_same"].tolist() == [1.0, 1.0]
    assert np.allclose(out["lift"], 2.0)

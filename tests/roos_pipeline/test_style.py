"""Figure 8's steps: cleaning, French by stopwords, masking, fragments, the guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from roos_pipeline import figures
from roos_pipeline.aggregate import (
    fragments,
    french_messages,
    mask_content_words,
    permutation_spearman,
)
from roos_pipeline.features import (
    FrenchByStopwords,
    StripWhatsAppNoise,
    french_share,
    unique_stopwords,
)


def test_noise_goes_and_every_apostrophe_is_straight():
    data = pd.DataFrame(
        {
            "message": [
                "c’est ça https://x.fr",
                "afbeelding weggelaten",
                "<Dit bericht is bewerkt> oui",
            ]
        }
    )
    out = StripWhatsAppNoise()(data)
    assert out["clean_message"].tolist() == ["c'est ça", "", "oui"]
    assert len(out) == 3  # an empty text stays a row: features keep the grain


def test_a_word_in_two_lists_says_nothing():
    unique = unique_stopwords()
    assert "de" not in unique["fr"]  # also Dutch
    assert "est" in unique["fr"]


def test_french_share_counts_only_recognised_stopwords():
    unique = unique_stopwords()
    # "c", "est", "pas" are French, "the" English: 3 of 4
    assert french_share("c'est pas the", unique) == pytest.approx(3 / 4)
    assert np.isnan(french_share("xyz", unique))


def test_french_by_stopwords_uses_the_threshold():
    data = pd.DataFrame({"clean_message": ["je suis pas là", "the cat is here", ""]})
    out = FrenchByStopwords()(data)
    assert out["is_french"].tolist() == [True, False, False]


def test_french_messages_drops_empty_texts_and_repeats():
    when = pd.Timestamp("2025-07-01 12:00")
    data = pd.DataFrame(
        {
            "ts_local": [when, when, when],
            "author": ["Roos"] * 3,
            "clean_message": ["oui c'est ça", "oui c'est ça", ""],
            "is_french": [True, True, True],
        }
    )
    assert french_messages(data)["message"].tolist() == ["oui c'est ça"]


def test_content_words_become_x_function_words_stay():
    assert mask_content_words("je mange une pomme ?") == "je x une x ?"


def test_fragments_drop_a_short_remainder():
    words = " ".join(["mot"] * 23)  # 15 + 8: two fragments; 15 + 5 would be one
    units = pd.DataFrame(
        {
            "author": ["Roos"],
            "ts_local": [pd.Timestamp("2025-07-01")],
            "message": [words],
        }
    )
    frags = fragments(units)
    assert frags["text"].str.split().map(len).tolist() == [15, 8]
    assert (frags["masked"].str.split().str[0] == "x").all()
    assert (frags["quarter"] == "2025Q3").all()


def test_permutation_spearman_finds_a_perfect_trend():
    rho, p = permutation_spearman(np.arange(8), np.arange(8) ** 2, n=500)
    assert rho == pytest.approx(1.0)
    assert p < 0.01


def test_style_guard_stops_a_stale_claim():
    trend = {"rho": 0.9, "p": 0.001, "n_months": 20}
    stats = {
        "roos_trend": trend,
        "louis_trend": {**trend, "rho": 0.5},  # Louis moving the wrong way
        "similarity_rho": 0.8,
        "similarity_p": 0.01,
    }
    with pytest.raises(ValueError, match="Louis"):
        figures.frans_van_roos_en_louis(pd.DataFrame(), pd.DataFrame(), stats)


def test_home_guard_stops_a_stale_claim():
    unit = {"gap_observed": 0.4, "p_value_greater": 0.0001}
    stats = {"messages": unit, "days": {**unit, "p_value_greater": 0.2}}
    with pytest.raises(ValueError, match="per days"):
        figures.onderweg_naar_thuis(pd.DataFrame(), stats)

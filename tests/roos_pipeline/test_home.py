"""Figure 7's steps: a gesprekje is the message with the place and the next 3."""

from __future__ import annotations

import pandas as pd
import pytest

from roos_pipeline.aggregate import home_gap, place_talks, theme_shares
from roos_pipeline.features import WithinNext, build_unmerged_feature_pipeline


def chat(*texts: str) -> pd.DataFrame:
    """Messages one minute apart, in time order, as `1_unmerged` holds them."""
    times = pd.date_range("2025-06-01", periods=len(texts), freq="min")
    return pd.DataFrame({"ts_local": times, "author": "Roos", "message": list(texts)})


def features_of(*texts: str) -> pd.DataFrame:
    return build_unmerged_feature_pipeline().apply(chat(*texts))


def test_a_theme_up_to_three_messages_later_counts():
    shares = theme_shares(
        features_of("paris demain", "ok", "oui", "le train part à 9h")
    )
    assert shares.loc["Parijs", "reizen"] == 1.0


def test_a_theme_four_messages_later_does_not():
    shares = theme_shares(
        features_of("paris demain", "ok", "oui", "top", "le train part à 9h")
    )
    assert shares.loc["Parijs", "reizen"] == 0.0


def test_a_theme_before_the_place_does_not():
    shares = theme_shares(features_of("le train part à 9h", "paris demain"))
    assert shares.loc["Parijs", "reizen"] == 0.0


def test_places_and_themes_ignore_case():
    shares = theme_shares(features_of("PARIS en TRAIN"))
    assert shares.loc["Parijs", "reizen"] == 1.0


def test_share_is_per_message_with_the_place_and_thuis_score_compares_the_groups():
    shares = theme_shares(features_of("paris en train", "paris, la maison"))
    assert shares.loc["Parijs", "berichten"] == 2
    # message 1's gesprekje holds message 2's "maison" too
    assert shares.loc["Parijs", "reizen"] == 0.5
    assert shares.loc["Parijs", "wonen"] == 1.0
    assert shares.loc["Parijs", "thuis_score"] == pytest.approx(1.0 / 1.5)


def test_place_talks_sum_to_the_thuis_score_and_skip_two_places():
    two = features_of("paris en train", "paris, la maison")
    talks = place_talks(
        features_of("paris en train", "paris, la maison", "paris ou utrecht?")
    )
    assert len(talks) == 2  # the message naming both places is left out
    paris = talks[talks["place"] == "Parijs"]
    expected = theme_shares(two).loc["Parijs", "thuis_score"]
    assert paris["home"].sum() / paris[["home", "away"]].sum().sum() == pytest.approx(
        expected
    )


def test_home_gap_is_place_minus_the_others_together():
    talks = pd.DataFrame(
        {"place": ["N", "N", "P", "A"], "home": [1, 1, 0, 1], "away": [0, 0, 1, 1]}
    )
    assert home_gap(talks, "N", ("P", "A")) == pytest.approx(1.0 - 1 / 3)


def test_within_next_refuses_rows_out_of_time_order():
    data = chat("a", "b").iloc[::-1].assign(flag=[True, False])
    with pytest.raises(ValueError, match="time order"):
        WithinNext(column="flag", window=3, feature="ahead")(data)

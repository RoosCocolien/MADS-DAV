"""Unit tests: one pipeline step at a time, on a few hand-made rows.

These catch *wrong numbers*: code that runs, passes ruff and ty, and still computes
the wrong thing. Whether a chart is *misleading* is not something a test can see;
that is what looking at the figure is for.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest
from goad_toolkit.datatransforms import Pipeline

from roos_pipeline import settings
from roos_pipeline.features import (
    LanguageFeature,
    MentionFlag,
    PregnancyWeek,
    normalise,
)
from roos_pipeline.prepare import CanonicalAuthor, LocalTimestamp, check_messages

# --- prepare -----------------------------------------------------------------------


def test_canonical_author_merges_aliases_only(aliases):
    data = pd.DataFrame(
        {"author": ["Louis Ancien", "~Louis Chessé", "Louis", "Louise"]}
    )
    out = Pipeline().add(CanonicalAuthor, aliases=aliases).apply(data)
    # An explicit map, not `"louis" in name`: a future "Louise" stays herself.
    assert out["author"].tolist() == ["Louis", "Louis", "Louis", "Louise"]


def test_check_messages_refuses_a_third_author():
    data = pd.DataFrame(
        {"author": ["Roos", "Louis", "Louis 2nd phone"], "ts_local": pd.Timestamp(0)}
    )
    with pytest.raises(ValueError, match="Expected 2 authors"):
        check_messages(data)


def test_local_timestamp_keeps_the_wall_clock():
    # 23:30 on the phone must stay 23:30 on the same day, not shift into the next.
    data = pd.DataFrame({"timestamp": ["2024-08-29 23:30:00+00:00"]})
    out = Pipeline().add(LocalTimestamp).apply(data)
    assert out["ts_local"].iloc[0] == pd.Timestamp("2024-08-29 23:30")
    assert out["ts_local"].dt.tz is None


# --- features ----------------------------------------------------------------------


def test_short_messages_get_no_language():
    data = pd.DataFrame({"message": ["Ok", "Ik hou van je, tot vanavond", None]})
    out = (
        Pipeline()
        .add(LanguageFeature, column="message", languages=settings.LANGUAGES)
        .apply(data)
    )
    assert out["language"].tolist() == ["unknown", "nl", "unknown"]


def test_sports_counts_a_message_once():
    data = pd.DataFrame(
        {"climbing": [1.0, 0.0, 1.0, 0.0], "running": [0.0, 2.0, 1.0, 0.0]}
    )
    out = (
        Pipeline()
        .add(MentionFlag, columns=["climbing", "running"], feature="sports")
        .apply(data)
    )
    assert out["sports"].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_mention_flag_fails_loudly_on_a_missing_column():
    data = pd.DataFrame({"climbing": [1.0]})
    with pytest.raises(ValueError, match="do not exist"):
        MentionFlag(columns=["climbing", "runing"], feature="sports")(data)


@pytest.mark.parametrize(
    ("day_offset", "week"),
    [
        (pd.Timedelta(weeks=5) - pd.Timedelta(seconds=1), 4),  # last second of week 4
        (pd.Timedelta(weeks=5), 5),  # first moment of the hard weeks
        (pd.Timedelta(weeks=15) - pd.Timedelta(seconds=1), 14),  # last hard moment
        (pd.Timedelta(weeks=15), 15),  # no longer hard
        (-pd.Timedelta(days=1), -1),  # before the pregnancy counts backwards
    ],
)
def test_pregnancy_week_boundaries(day_offset, week):
    data = pd.DataFrame({"ts_local": [settings.PREGNANCY_START + day_offset]})
    out = PregnancyWeek(start=settings.PREGNANCY_START)(data)
    assert out["pregnancy_week"].iloc[0] == week


def test_hard_weeks_are_ten_and_start_after_the_test():
    assert len(settings.HARD_WEEKS) == 10
    hard_start = settings.PREGNANCY_START + pd.Timedelta(weeks=settings.HARD_WEEKS[0])
    assert hard_start > settings.TEST_DATE


# --- regexes: what each pattern must and must not match ----------------------------


def matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text) is not None


@pytest.mark.parametrize(
    ("text", "symptom", "expected"),
    [
        ("Je suis écœurée ce matin", "nausea", True),
        ("ik ben zo misselijk", "nausea", True),
        ("ik ben zo moe", "fatigue", True),
        ("so tired today", "fatigue", True),
        ("mijn moeder belt straks", "fatigue", False),  # 'moe' only as a whole word
        ("ce n'est pas nauséabond", "nausea", False),
    ],
)
def test_symptom_patterns(text, symptom, expected):
    assert matches(settings.SYMPTOM_PATTERNS[symptom], normalise(text)) is expected


@pytest.mark.parametrize(
    ("text", "topic", "expected"),
    [
        ("on va grimper ce soir", "climbing", True),
        ("I sent a 6a+ today", "climbing", True),
        ("what a surprise!", "climbing", False),
        ("il a une nouvelle entreprise", "climbing", False),
        ("poffertjes eten?", "climbing", False),
        ("Benji slaapt", "child", True),
        ("il y a des jouets partout", "child", False),
        ("on a déposé Benji chez Partou", "child", True),
        ("il faut changer les couches", "child", True),
        ("je vais me coucher", "child", False),
        ("l'accouchement s'est bien passé", "child", True),
        ("des prises toutes neuves", "climbing", True),
        ("I ran 10k", "running", True),
    ],
)
def test_topic_patterns(text, topic, expected):
    assert matches(settings.TOPIC_PATTERNS[topic], text) is expected

"""Shared test data: a fake chat small enough to know every right answer by hand.

The real chat is personal and gitignored, so the tests never touch it. A classmate
(or a hook) can run them on a fresh clone.
"""

from __future__ import annotations

import pandas as pd
import pytest

from roos_pipeline import settings


def message(when: object, author: str, text: str) -> dict:
    """One row shaped like the real export: the timestamp as a '+00:00' string.

    `when` is typed `object`: pandas' stubs say Timestamp + Timedelta may be NaT, and
    all this function does with it is put it in an f-string.
    """
    return {"timestamp": f"{when}+00:00", "author": author, "message": text}


@pytest.fixture
def fake_chat() -> pd.DataFrame:
    """Two people, three display names, every pregnancy week from -2 to 16.

    Weeks -2..3 ("normal"): one fatigue message a week. Weeks 5..14 ("hard"): three
    a week. Plus a few messages for language and topics, and one bare timestamp in
    week 4, which belongs to neither group.
    """
    start = settings.PREGNANCY_START
    rows = []
    for week in range(-2, 17):
        for day in range(7):
            when = start + pd.Timedelta(weeks=week, days=day, hours=12)
            rows.append(message(when, "Roos", "hallo hoe gaat het"))
        n_tired = 3 if week in settings.HARD_WEEKS else 1
        for i in range(n_tired):
            when = start + pd.Timedelta(weeks=week, days=1, hours=8 + i)
            rows.append(message(when, "Louis Ancien", "je suis crevée"))
    before_move = settings.MOVE_DATE - pd.Timedelta(days=2)
    rows += [
        message(before_move, "~Louis Chessé", "we go climbing tomorrow"),
        message(before_move, "Louis", "Ok"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def aliases() -> dict[str, str]:
    return {"Louis Ancien": "Louis", "~Louis Chessé": "Louis"}

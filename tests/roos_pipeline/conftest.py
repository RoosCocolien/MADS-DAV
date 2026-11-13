"""Shared test data: a fake chat small enough to know every right answer by hand.

The real chat is personal and gitignored, so the tests never touch it. A classmate
(or a hook) can run them on a fresh clone.
"""

from __future__ import annotations

import numpy as np
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

    Weeks -2..12 (all before the move): Louis also writes 5-7 two-word French
    messages and 6 eight-word English ones, on a Tuesday, so each lands in one
    calendar week. His French is 6 words shorter (median) in every one of them.

    After the birth: places with themes (figure 7, `place_rows`) and a year and a half
    of French from both (figure 8, `french_rows`).
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
    for week in range(-2, 13):
        tuesday = start + pd.Timedelta(weeks=week, days=2)
        n_french = 5 + week % 3  # a different French share from week to week
        for i in range(n_french):
            when = tuesday + pd.Timedelta(hours=9, minutes=i)
            rows.append(message(when, "Louis", "bonne nuit"))
        for i in range(6):
            when = tuesday + pd.Timedelta(hours=18, minutes=i)
            rows.append(
                message(when, "Louis", "that sounds like a lovely idea for saturday")
            )
    before_move = settings.MOVE_DATE - pd.Timedelta(days=2)
    rows += [
        message(before_move, "~Louis Chessé", "we go climbing tomorrow"),
        message(before_move, "Louis", "Ok"),
    ]
    return pd.DataFrame(rows + place_rows() + french_rows())


# After Benji's birth, so none of figures 1-6's periods or weeks see these rows.
PLACES_START = settings.BIRTH_DATE + pd.Timedelta(days=21)
FRENCH_START, FRENCH_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2026-12-31")

# One place a day, each message followed by three without a theme, so a gesprekje
# never reaches the next place: Parijs/Amsterdam/Utrecht only on the road, Nieuwegein
# only at home, Zoetermeer both. Figure 7's shuffle test must then find a clear gap.
PLACE_MESSAGES = (
    [("Roos", "ik neem de trein naar Parijs")] * 4
    + [("Louis", "Amsterdam: eerst klimmen, dan een biertje")] * 3
    + [("Roos", "in Utrecht naar het restaurant")] * 3
    + [("Louis", "Nieuwegein: Benji slaapt, wij zoeken een huis")] * 7
    + [("Roos", "met de trein naar Zoetermeer, oma past op Benji")]
)
FILLERS = ("ok", "top", "goed")

# French in four styles, after the axis markers of figure 8 (settings.STYLE_MARKERS).
SCHOOL = "je pense que nous avons beaucoup de livres est-ce que tu n'as pas le temps :)"
SPOKEN = "oui ca va il faut que tu viennes quoi on se voit ce soir ou pas ? je suis la"
LOOSE = "c'est cool ;) or pas : on verra demain ! je suis la pour toi ! bisous or pas"
CORRECT = (
    "il doit venir chez nous demain et je veux bien lire avec lui si il peut le faire"
)


def place_rows() -> list[dict]:
    rows = []
    for day, (author, text) in enumerate(PLACE_MESSAGES):
        when = PLACES_START + pd.Timedelta(days=day, hours=20)
        rows.append(message(when, author, text))
        for i, filler in enumerate(FILLERS, start=1):
            rows.append(message(when + pd.Timedelta(minutes=i), "Roos", filler))
    return rows


def french_rows() -> list[dict]:
    """One French message a day each, Jul 2025 - Dec 2026, growing towards each other.

    Early on Roos writes school French and Louis loose French with English leftovers;
    with time t (0 -> 1) each writes more of a shared style (spoken and correct). So
    Roos moves towards spoken French, Louis away from the loose style, and the two
    profiles become more alike: the three claims under figure 8.
    """
    rng = np.random.default_rng(settings.SEED)
    days = pd.date_range(FRENCH_START, FRENCH_END, freq="D")
    rows = []
    for i, day in enumerate(days):
        t = i / (len(days) - 1)
        for author, hour, own in (("Roos", 12, SCHOOL), ("Louis", 13, LOOSE)):
            text = rng.choice([SPOKEN, CORRECT]) if rng.random() < t else own
            rows.append(message(day + pd.Timedelta(hours=hour), author, str(text)))
    return rows


@pytest.fixture
def aliases() -> dict[str, str]:
    return {"Louis Ancien": "Louis", "~Louis Chessé": "Louis"}

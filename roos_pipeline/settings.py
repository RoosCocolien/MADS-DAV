"""Every date, regex, colour and path the pipeline uses, in one place.

Before this module existed the notebooks disagreed with each other: the positive test
was 2024-09-01 in roos_03.3 and 2024-08-29 in roos_04.6, Benji's birth 2025-05-01 in
scripts/chat_pipeline.py and 2025-05-11 everywhere else. A pipeline can only have one
answer, so the answer lives here and every stage imports it (decided 2026-10-06, see
analysis-log.md).

Paths are a dataclass rather than constants, so a test can point the whole pipeline at
a temporary folder (`Paths(root=tmp_path)`) without touching the real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from wa_analyzer.data import ROOT

# --- the story's timeline: Roos's own dates, not derived from the data -------------

MOVE_DATE = pd.Timestamp("2024-11-01")  # Louis moves to Utrecht (day unknown, the 1st)
TEST_DATE = pd.Timestamp("2024-08-29")  # positive pregnancy test
BIRTH_DATE = pd.Timestamp("2025-05-11")  # Benji is born
# Due-date convention: pregnancy week n starts at PREGNANCY_START + n weeks.
PREGNANCY_START = BIRTH_DATE - pd.Timedelta(weeks=41)

# The hard weeks of the pregnancy (5 up to the start of 15 = ten weeks), and the
# long-distance weeks before the pregnancy that serve as "normal".
HARD_WEEKS = range(5, 15)
LAST_NORMAL_WEEK = 3  # weeks < 4; week 4 holds the test and is left out

# --- language ----------------------------------------------------------------------

LANGUAGES = ["en", "nl", "fr"]
MIN_CHARS = 3  # shorter messages ("Ok", one emoji) are no evidence of a language
# A reply that quotes an earlier message carries its "[18-12-2025, 09:42:27] Roos: ..."
# line, so the quoted words count as the sender's (found in roos_05.6, 2026-10-09).
QUOTED_REPLY_PATTERN = r"\[\d{1,2}-\d{1,2}-\d{4}, "

# --- topics: the roos_03.3 versions (chosen 2026-10-06 over the narrower 01.4 ones) --
# Three word boundaries added on 2026-10-06, after a test caught false matches on the real
# chat: `prise`/`pof` matched surprise (133x), entreprise (25x), poffertjes (5x);
# `partou` matched partout (23x); `couches?` matched coucher (47x) and accouchement.
# `accouchement` (childbirth) then got its own word, so it still counts as child.

TOPIC_PATTERNS: dict[str, str] = {
    "child": (
        r"(?i)("
        r"benji|b[ée]b[ée]|\bbaby\b|cr[eè]che|kinderdagverblijf|opvang|daycare|partou\b|basker|kinderdagopvang|"
        r"r[eé]cuperer|r[eé]cupes|prendre soin de|"
        r"\benfant\b|biberon|bottle[- ]?feeding|sucette|fopspeen|speentje|"
        r"pacifier|dummy|allaiter|donner les? seins?|borstvoeding|breastfeeding|t[ée]t[ée]e|"
        r"luiers?|diapers?|couches?\b|doekje|hydrophile|vitamines|hydrofiele doek|\bwipe\b|"
        r"kinderwagen|poussette|stroller|pram|la remorque|bike trailer|"
        r"consultatiebureau|kraamzorg|zwangerschap|grossesse|pregnan(cy|t)|"
        r"postnataal|zwanger|enceinte|accouchements?|"
        r"verloskundige|sage-femme|echo|midwife|kolven|[ée]chographie"
        r")"
    ),
    "climbing": (
        r"(?i)("
        r"escalad(er|e)|salle de grimpe|grimpe|grimper|\bclimb(ing)?\b|climbing gym|"
        r"klimmen|klimmuur( centraal)?|\bboulder\b|\bbloc\b|\bvoie\b|"
        r"arkose|mountain network|\bncc\b|neoliet|\b[4-8][abc]\+?\b|"
        r"national climbing center|\bpof\b|magnesium|\bprises?\b"
        r")"
    ),
    "running": (
        r"(?i)("
        r"\brun(ning)?\b|courir|\btrail\b|\bbornes?\b|kilom[eè]tre|\bkm\b|marathon|"
        r"hardlopen|hardloopschoenen|joggen|strava|parkrun|\b10k\b|\btempo\b|"
        r"chaussures|\bshoes\b"
        r")"
    ),
}

# --- symptoms: run on normalised text (lowercase, no accents), see roos_04.6c ------

SYMPTOM_PATTERNS: dict[str, str] = {
    "nausea": (
        r"\b(?:"
        r"nausees?|nauseeuses?|nauseeux|(?:envie de )?vomir|ecoeuree?s?|barbouillee?s?|"
        r"misselijk(?:heid)?|overgeven|kotsen|braken|"
        r"nausea|nauseous|vomit(?:ing|ed)?|throw(?:ing)? up|feel(?:ing)? sick|morning sickness"
        r")\b"
    ),
    "fatigue": (
        r"\b(?:"
        r"fatiguee?s?|crevee?s?|epuisee?s?|naze|claquee?s?|j'?\s?ai sommeil|pas d'energie|"
        r"moe|vermoeid|uitgeput|doodop|slaperig|geen energie|"
        r"tired|exhausted|sleepy|knackered|drained|worn out|no energy"
        r")\b"
    ),
}

# --- statistics --------------------------------------------------------------------

SMOOTH_WEEKS = 4  # trailing rolling average in "Baby is taking over", as in roos_03.3
N_SHUFFLE = 10_000
RUN_WEEKS = 10  # time-order check: runs of this many consecutive weeks, as in 04.6d
SEED = 7

# --- colours (grey-first, 02.2; Okabe-Ito where a colour carries meaning) ----------

GREY = "#7f7f7f"
LIGHT_GREY = "#C8C8C8"
CRIMSON = "crimson"  # goad's PlotSettings().highlight_color
CHILD_BLUE = "#0072B2"
SPORT_ORANGE = "#D55E00"
# roos_05.6, French vs. message length: Okabe-Ito orange and blue (Roos's pick, 2026-10-09)
LONG_DISTANCE_ORANGE = "#E69F00"
TOGETHER_BLUE = "#0072B2"
DARK_GREY = "#333333"  # a fitted line over coloured dots

PERIOD_ORDER = ["before move", "after move"]
PERIOD_LABELS_NL = {
    "before move": "Voor de verhuizing",
    "after move": "Na de verhuizing",
}


@dataclass(frozen=True)
class Paths:
    """Where each stage reads and writes. Every stage boundary is a file here."""

    root: Path = ROOT

    @property
    def processed(self) -> Path:
        # Gitignored (data/processed/**): the chat itself never ends up in git.
        return self.root / "data" / "processed" / "roos"

    @property
    def figures(self) -> Path:
        return self.root / "figures" / "pipeline"

    @property
    def config(self) -> Path:
        return self.root / "config.toml"

    @property
    def messages(self) -> Path:
        return self.processed / "1_messages.parquet"

    @property
    def features(self) -> Path:
        return self.processed / "2_features.parquet"

    @property
    def language_share(self) -> Path:
        return self.processed / "3_language_share.parquet"

    @property
    def weekly_topics(self) -> Path:
        return self.processed / "3_weekly_topics.parquet"

    @property
    def topic_stats(self) -> Path:
        return self.processed / "3_topic_stats.json"

    @property
    def weekly_symptoms(self) -> Path:
        return self.processed / "3_weekly_symptoms.parquet"

    @property
    def symptom_stats(self) -> Path:
        return self.processed / "3_symptom_stats.json"

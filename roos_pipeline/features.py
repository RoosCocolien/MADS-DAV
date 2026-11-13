"""Stage 2 — features: add columns, never change the grain (one row per message stays one).

    1_messages.parquet
      -> LanguageFeature                  language            (Praat Nederlands)
      -> SplitAtDate                      move_period         (Praat Nederlands)
      -> RegexFeature x3 -> MentionFlag   child, climbing, running, sports
                                                              (Baby is taking over)
      -> NormaliseText -> RegexFeature x2 -> MentionFlag
                                          nausea, fatigue, either  (prepare and plan)
      -> RegexFeature -> MapValues        n_words, is_french  (familiar language)
    -> 2_features.parquet

    1_unmerged.parquet
      -> RegexFeature x5                  place_<city>        (onderweg naar thuis)
      -> RegexFeature -> WithinNext x6    theme_<theme>, talk_<theme>
      -> StripWhatsAppNoise               clean_message       (style map)
      -> FrenchByStopwords                french_share, is_french
    -> 2_unmerged_features.parquet

`RegexFeature` and `MapValues` come from goad_toolkit as-is; the other classes are new,
and every one is a `TransformBase` so it slots into `Pipeline.add(...)` like goad's own
steps.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
import py3langid as langid
from goad_toolkit.datatransforms import (
    MapValues,
    Pipeline,
    RegexFeature,
    TransformBase,
)
from loguru import logger

from roos_pipeline import settings
from roos_pipeline.settings import Paths


class LanguageFeature(TransformBase):
    """Detect the language of a text column with `py3langid`.

    Restricting the candidate set (`languages`) is what makes this usable on short,
    sometimes-incomplete chat messages: without it, `py3langid` picks among ~97
    languages and misfires far more often on a five-word message. Messages under
    `min_chars` become "unknown": an "Ok" is no evidence of a language either way.
    """

    def transform(
        self,
        data: pd.DataFrame,
        column: str,
        languages: list[str],
        feature: str = "language",
        min_chars: int = settings.MIN_CHARS,
    ) -> pd.DataFrame:
        langid.set_languages(languages)

        def detect(msg: str) -> str:
            if len(msg.strip()) < min_chars:
                return "unknown"
            return langid.classify(msg)[0]

        data[feature] = data[column].fillna("").map(detect)
        return data


class SplitAtDate(TransformBase):
    """Label each message as falling before or after one date."""

    def transform(
        self,
        data: pd.DataFrame,
        date: pd.Timestamp,
        labels: tuple[str, str] = ("before", "after"),
        column: str = "ts_local",
        feature: str = "period",
    ) -> pd.DataFrame:
        data[feature] = np.where(data[column] < date, labels[0], labels[1])
        return data


class MentionFlag(TransformBase):
    """1.0 if any of `columns` is non-zero (or True), else 0.0.

    One class for two jobs. With one column it turns a word count into "does this
    message mention the topic at all"; the mean of that flag over a week is the
    topic's share of the week. With several columns it is an OR: `sports` = climbing
    OR running, so a message that mentions both still counts once.

    A float rather than a bool, because the next stage averages it.
    """

    def transform(
        self, data: pd.DataFrame, columns: list[str], feature: str
    ) -> pd.DataFrame:
        missing = [c for c in columns if c not in data.columns]
        if missing:
            raise ValueError(f"{self.name}: columns {missing} do not exist yet")
        data[feature] = data[columns].astype(float).gt(0).any(axis=1).astype(float)
        return data


def normalise(text: str) -> str:
    """Lowercase, curly apostrophes straightened, accents removed: 'Écœurée' -> 'ecoeuree'."""
    text = str(text).lower().replace("’", "'").replace("œ", "oe")
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


class NormaliseText(TransformBase):
    """A copy of a text column passed through `normalise`, so the symptom patterns can
    be written without accents and still match 'écœurée' and 'ecoeuree' alike."""

    def transform(
        self, data: pd.DataFrame, column: str, feature: str = "normalised"
    ) -> pd.DataFrame:
        data[feature] = data[column].fillna("").map(normalise)
        return data


class PregnancyWeek(TransformBase):
    """Which pregnancy week a row's date falls in: week n starts at `start` + n weeks.

    Negative before the pregnancy, and it keeps counting after the birth: deciding
    which weeks are "hard" or "normal" is a separate step. Used by the aggregate stage
    on one row per calendar day, not here on messages: the weekly comparison needs the
    days without any message too, and a message-level column cannot have those.
    """

    def transform(
        self,
        data: pd.DataFrame,
        start: pd.Timestamp,
        column: str = "ts_local",
        feature: str = "pregnancy_week",
    ) -> pd.DataFrame:
        data[feature] = (data[column].dt.normalize() - start).dt.days // 7
        return data


class WithinNext(TransformBase):
    """True where `column` is True in this row or in one of the `window` rows after it.

    The "gesprekje" of figure 7: a theme belongs to a message naming a place when it
    comes up there or in the next `window` messages, whoever writes them. Rows must be
    in time order, which `1_unmerged` is (UnmergeMessages sorts it).
    """

    def transform(
        self, data: pd.DataFrame, column: str, window: int, feature: str
    ) -> pd.DataFrame:
        if not data["ts_local"].is_monotonic_increasing:
            raise ValueError(f"{self.name}: rows must be in time order")
        hit = data[column].to_numpy(dtype=bool)
        ahead = hit.copy()
        for k in range(1, window + 1):
            ahead[:-k] |= hit[k:]
        data[feature] = ahead
        return data


class StripWhatsAppNoise(TransformBase):
    """`feature`: the text without WhatsApp's system lines and URLs, every apostrophe '.

    A row whose text is only noise keeps an empty string, so the grain stays one row
    per message; the aggregate stage leaves those out (an empty text is never French).
    """

    def transform(
        self,
        data: pd.DataFrame,
        column: str = "message",
        feature: str = "clean_message",
    ) -> pd.DataFrame:
        text = data[column].fillna("").str.replace("‎", "", regex=False)
        for pattern in (*settings.STYLE_SYSTEM_PATTERNS, settings.LINK_PATTERN):
            text = text.str.replace(pattern, " ", regex=True, flags=re.IGNORECASE)
        quotes = settings.STYLE_APOSTROPHES
        text = text.str.translate(str.maketrans(quotes, "'" * len(quotes)))
        data[feature] = text.str.replace(r"\s+", " ", regex=True).str.strip()
        return data


def unique_stopwords(
    stopwords: dict[str, frozenset[str]] = settings.STYLE_STOPWORDS,
) -> dict[str, frozenset[str]]:
    """Per language, only the words that are in no other language's list."""
    return {
        language: words.difference(
            *(other for name, other in stopwords.items() if name != language)
        )
        for language, words in stopwords.items()
    }


def french_share(text: str, unique: dict[str, frozenset[str]]) -> float:
    """Share of the recognised (unique) stopwords that are French; NaN if none is."""
    counts = dict.fromkeys(unique, 0)
    for word in re.findall(settings.STYLE_LETTER_RUN, text.lower()):
        for language, words in unique.items():
            if word in words:
                counts[language] += 1
    recognised = sum(counts.values())
    return counts["fr"] / recognised if recognised else float("nan")


class FrenchByStopwords(TransformBase):
    """`is_french`: at least `threshold` of the recognised stopwords are French.

    Figure 8's own language rule (roos_06.6 §6.6.7), next to `language` from
    py3langid: hand-made stopword lists, only the words unique to one language.
    """

    def transform(
        self,
        data: pd.DataFrame,
        column: str = "clean_message",
        threshold: float = settings.STYLE_FRENCH_SHARE,
    ) -> pd.DataFrame:
        unique = unique_stopwords()
        data["french_share"] = [french_share(text, unique) for text in data[column]]
        data["is_french"] = data["french_share"] >= threshold
        return data


def yes_no_pattern(pattern: str) -> str:
    """A pattern ready for `RegexFeature(mode="has")`: case-insensitive, no groups.

    `RegexFeature` matches case-sensitively, and "(?i)" must stand at the very start.
    The lesson-3 topic patterns have capture groups, which `str.contains` warns about;
    a yes/no needs none, so every unescaped "(" that is not already "(?" becomes "(?:".
    """
    pattern = re.sub(r"(?<!\\)\((?!\?)", "(?:", pattern.removeprefix("(?i)"))
    return "(?i)" + pattern


def build_feature_pipeline() -> Pipeline:
    """Every per-message column the six figures need, in dependency order."""
    pipeline = Pipeline()
    pipeline.add(LanguageFeature, column="message", languages=settings.LANGUAGES)
    pipeline.add(
        SplitAtDate,
        name="move_period",
        date=settings.MOVE_DATE,
        labels=("before move", "after move"),
        feature="move_period",
    )
    for topic, pattern in settings.TOPIC_PATTERNS.items():
        pipeline.add(
            RegexFeature,
            name=f"{topic}_words",
            column="message",
            pattern=pattern,
            feature=f"n_{topic}_words",
            mode="count",
        )
        pipeline.add(
            MentionFlag,
            name=f"{topic}_flag",
            columns=[f"n_{topic}_words"],
            feature=topic,
        )
    pipeline.add(
        MentionFlag,
        name="sports_flag",
        columns=["climbing", "running"],
        feature="sports",
    )
    pipeline.add(NormaliseText, column="message")
    for symptom, pattern in settings.SYMPTOM_PATTERNS.items():
        pipeline.add(
            RegexFeature,
            name=f"{symptom}_has",
            column="normalised",
            pattern=pattern,
            feature=symptom,
            mode="has",
        )
    pipeline.add(
        MentionFlag,
        name="either_flag",
        columns=list(settings.SYMPTOM_PATTERNS),
        feature="either",
    )
    pipeline.add(
        RegexFeature,
        name="words",
        column="message",
        pattern=settings.WORD_PATTERN,
        feature="n_words",
        mode="count",
    )
    # NaN for "unknown": a weekly mean then skips it, so the French share is a share
    # of the classified messages only, like the bars of figures 1 and 2.
    pipeline.add(
        MapValues,
        column="language",
        mapping={"fr": 1.0, "en": 0.0, "nl": 0.0},
        feature="is_french",
    )
    return pipeline


def build_unmerged_feature_pipeline() -> Pipeline:
    """The per-message columns of figures 7 and 8, on `1_unmerged`.

    Figure 7: a flag per place and per theme (goad's `RegexFeature`), and per theme
    whether it comes up in the message's gesprekje (`WithinNext`). Figure 8: the text
    without WhatsApp's noise, and whether it counts as French by stopwords.
    """
    pipeline = Pipeline()
    for place, pattern in settings.HOME_PLACES.items():
        pipeline.add(
            RegexFeature,
            name=f"place_{place}",
            column="message",
            pattern=yes_no_pattern(pattern),
            feature=f"place_{place}",
            mode="has",
        )
    for theme, (_, pattern) in settings.HOME_THEMES.items():
        pipeline.add(
            RegexFeature,
            name=f"theme_{theme}",
            column="message",
            pattern=yes_no_pattern(pattern),
            feature=f"theme_{theme}",
            mode="has",
        )
        pipeline.add(
            WithinNext,
            name=f"talk_{theme}",
            column=f"theme_{theme}",
            window=settings.HOME_WINDOW,
            feature=f"talk_{theme}",
        )
    pipeline.add(StripWhatsAppNoise, column="message")
    pipeline.add(FrenchByStopwords, column="clean_message")
    return pipeline


def run(paths: Paths) -> pd.DataFrame:
    """Read both stage-1 files, add every feature, write both stage-2 files."""
    messages = pd.read_parquet(paths.messages)
    features = build_feature_pipeline().apply(messages)
    features.to_parquet(paths.features)
    unmerged = build_unmerged_feature_pipeline().apply(pd.read_parquet(paths.unmerged))
    unmerged.to_parquet(paths.unmerged_features)
    logger.success(
        f"features: {features.shape[1]} columns on {len(features):,} messages "
        f"-> {paths.features}; {unmerged.shape[1]} columns on {len(unmerged):,} "
        f"split messages -> {paths.unmerged_features}"
    )
    return features

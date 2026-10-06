"""Stage 2 — features: add columns, never change the grain (one row per message stays one).

    1_messages.parquet
      -> LanguageFeature                  language            (Praat Nederlands)
      -> SplitAtDate                      move_period         (Praat Nederlands)
      -> RegexFeature x3 -> MentionFlag   child, climbing, running, sports
                                                              (Baby is taking over)
      -> NormaliseText -> RegexFeature x2 -> MentionFlag
                                          nausea, fatigue, either  (prepare and plan)
    -> 2_features.parquet

`RegexFeature` comes from goad_toolkit as-is; the other classes are new, and every one
is a `TransformBase` so it slots into `Pipeline.add(...)` like goad's own steps.
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd
import py3langid as langid
from goad_toolkit.datatransforms import Pipeline, RegexFeature, TransformBase
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


def build_feature_pipeline() -> Pipeline:
    """Every per-message column the five figures need, in dependency order."""
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
    return pipeline


def run(paths: Paths) -> pd.DataFrame:
    """Read 1_messages.parquet, add every feature, write 2_features.parquet."""
    messages = pd.read_parquet(paths.messages)
    features = build_feature_pipeline().apply(messages)
    features.to_parquet(paths.features)
    logger.success(
        f"features: {features.shape[1]} columns on {len(features):,} messages "
        f"-> {paths.features}"
    )
    return features

"""The topic word lists (`settings.ACTIVITY_TOPICS`): the false friends stay out."""

from __future__ import annotations

import re

import pytest

from scripts import roos_settings as settings


def topics_in(text: str) -> set[str]:
    return {
        topic
        for topic, (_, pattern) in settings.ACTIVITY_TOPICS.items()
        if re.search(pattern, text, re.IGNORECASE)
    }


@pytest.mark.parametrize(
    ("text", "not_this"),
    [
        ("je suis dans la cuisine", "dansen"),
        ("of course we can combine", "studie & school"),
        ("les étudiantes", "studie & school"),
        ("comment s'est passé ta soirée", "feest & borrel"),
        ("tu peux prendre une photo?", "fotografie"),
        ("we moeten de trein nog boeken", "lezen & boeken"),
    ],
)
def test_a_false_friend_is_not_a_topic(text: str, not_this: str):
    assert not_this not in topics_in(text)


@pytest.mark.parametrize(
    ("text", "topic"),
    [
        ("ce soir je vais danser la salsa", "dansen"),
        ("mon examen est demain", "studie & school"),
        ("raclette party!", "feest & borrel"),
        ("je veux lire ce livre", "lezen & boeken"),
        ("une séance de yoga", "yoga"),
    ],
)
def test_the_topic_itself_is_found(text: str, topic: str):
    assert topic in topics_in(text)

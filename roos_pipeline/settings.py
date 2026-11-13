"""Every date, regex, colour and path the six figures need, in one place, sorted by kind.

Before this module existed the notebooks disagreed with each other: the positive test
was 2024-09-01 in roos_03.3 and 2024-08-29 in roos_04.6, Benji's birth 2025-05-01 in
scripts/chat_pipeline.py and 2025-05-11 everywhere else. A pipeline can only have one
answer, so the answer lives here and every stage imports it (decided 2026-10-06, see
analysis-log.md).

Since 2026-11-13 the settings are pydantic classes, one per kind (Roos's choice, "per
soort"), instead of ~150 loose constants:

    Timeline    when: the story's dates and pregnancy weeks
    Categories  the names of the groups: authors, languages, periods, topics
    Patterns    every regex
    WordLists   hand-made lists of words and characters
    Limits      thresholds: what counts, and what is left out
    Grouping    how messages are cut into windows, pieces and periods
    Tests       the shuffle tests (one class per test, all inheriting `ShuffleTest`)
    Colors      one palette per figure (all inheriting `Palette`)
    Paths       where each stage reads and writes

A function that needs a setting takes the class as a parameter, e.g.
`weekly_french(louis, limits: Limits = LIMITS, ...)`, so its signature says which kind of
settings it uses, and a test can pass its own (`ShuffleTest(n_shuffle=200)`).

The notebooks keep their own flat settings in `scripts/roos_settings.py`, which takes the
shared values (dates, regexes, colours) from here.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from pydantic import BaseModel, ConfigDict

from wa_analyzer.data import ROOT


class Frozen(BaseModel):
    """The base of every settings class: read-only once made, pandas types allowed."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


# --- named types used in the settings ----------------------------------------------


class Side(StrEnum):
    """The two groups of themes in les 7."""

    AWAY = "onderweg"
    HOME = "thuis"


class Labelled(NamedTuple):
    """A regex with the group it belongs to, e.g. ("thuis", r"\\bhuis\\b").

    A NamedTuple is still a tuple, so `group, pattern = theme` and `theme[1]` keep
    working; `theme.pattern` says what the [1] meant.
    """

    group: str
    pattern: str


# --- Timeline: Roos's own dates, not derived from the data ---------------------------


class Timeline(Frozen):
    move: pd.Timestamp = pd.Timestamp("2024-11-01")  # Louis moves to Utrecht (the 1st)
    test: pd.Timestamp = pd.Timestamp("2024-08-29")  # positive pregnancy test
    birth: pd.Timestamp = pd.Timestamp("2025-05-11")  # Benji is born
    # The hard weeks of the pregnancy (5 up to the start of 15 = ten weeks), and the
    # long-distance weeks before the pregnancy that serve as "normal".
    hard_weeks: range = range(5, 15)
    last_normal_week: int = 3  # weeks < 4; week 4 holds the test and is left out

    @property
    def pregnancy_start(self) -> pd.Timestamp:
        """Due-date convention: pregnancy week n starts here + n weeks."""
        return self.birth - pd.Timedelta(weeks=41)


# --- Categories: the names of the groups the data is split into ----------------------


class Categories(Frozen):
    authors: tuple[str, str] = ("Roos", "Louis")
    languages: tuple[str, ...] = ("en", "nl", "fr")
    periods: tuple[str, str] = ("before move", "after move")
    # The two topics drawn in "Baby is taking over" (les 3).
    trend_topics: tuple[str, str] = ("child", "climbing")


# --- Patterns: every regex -----------------------------------------------------------


class Patterns(Frozen):
    # "‎[13-08-2023, 14:37:06] Roos: " in the middle of a row: the start of another message.
    embedded_header: str = (
        r"‎?\[(\d{1,2}-\d{1,2}-\d{4}, \d{2}:\d{2}:\d{2})\] ([^:\n]+): "
    )
    # Words are runs of characters without a space: an emoji or a link is one word.
    word: str = r"\S+"
    link: str = r"https?://\S+|www\.\S+"

    # Topics: the roos_03.3 versions (chosen 2026-10-06 over the narrower 01.4 ones).
    # Three word boundaries added on 2026-10-06, after a test caught false matches on the
    # real chat: `prise`/`pof` matched surprise (133x), entreprise (25x), poffertjes (5x);
    # `partou` matched partout (23x); `couches?` matched coucher (47x) and accouchement.
    # `accouchement` (childbirth) then got its own word, so it still counts as child.
    # Running is drawn nowhere on its own: les 7's "klimmen & sport" reuses it.
    topics: dict[str, str] = {
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

    # Symptoms: run on normalised text (lowercase, no accents), see roos_04.6c.
    symptoms: dict[str, str] = {
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

    # Les 7, from on the road to home (Roos's rules, 2026-11-12). Five cities, Dutch
    # names; Paris includes Gare du Nord, which is also a "reizen" word, so a Gare du
    # Nord message always counts as travel too.
    home_places: dict[str, str] = {
        "Parijs": r"\bparis\b|\bparijs\b|\bgare du nord\b",
        "Amsterdam": r"\bamsterdam(?:se)?\b",
        "Utrecht": r"\butrecht?\b",
        "Zoetermeer": r"\bzoetermeer\b",
        "Nieuwegein": r"\bnieuwegein\b",
    }
    # The activity word lists of roos_07.4 §7.4.12 that les 7's themes reuse (checked on
    # samples, 2026-11-03; the notebook's full list lives in scripts/roos_settings.py).
    activities: dict[str, str] = {
        "fietsen": r"\bfiets(?:en|tocht)?\b|\bv[ée]lo\b|\bcycling\b|\bbike ride\b|\bwielren",
        "zwemmen": r"\bzwem(?:men|bad)?\b|\bnager\b|\bnatation\b|\bpiscine\b|\bswim(?:ming)?\b",
        "yoga": r"\byoga\b",
        "wandelen": r"\bwandel(?:en|ing|tocht)?\b|\brandonn[ée]e\b|\bhik(?:e|ing)\b|\bbalade à pied\b|\bpromenade\b|\bwalk(?:ing)?\b",
        "uit eten": r"\brestaurant\b|\bresto\b|\buit eten\b|\bbrunch\b|\bterras(?:je)?\b",
        "feest & borrel": r"\bf[eê]tes?\b|\bfeest(?:je)?\b|\bparty\b|\bborrel\b|\bap[ée]ro\b|\bdrinks\b",
    }
    # The words of the four themes that are no topic or activity above.
    theme_words: dict[str, str] = {
        "reizen": (
            r"\btrein\b|\btrains?\b|\btgv\b|\bthalys\b|\beurostar\b|\bouigo\b|\bsncf\b|\bgare\b"
            r"|\bstation\b|\bperron\b|\bquai\b|\bretard\b|\bvertraging\b|\bdelay(?:ed)?\b"
            r"|\bbillets?\b|\btickets?\b|\bkaartje\b|\bvlucht\b|\bflight\b|\bvliegtuig\b"
            r"|\bavion\b|\bblablacar\b|\bcovoiturage\b"
        ),
        "eten & uitgaan": r"\bbar\b|\bd[iî]ner\b|\bdinner\b|\blunch\b|\bbiertje\b|\bbi[eè]re\b|\bbeer\b",
        "wonen": (
            r"\bhuis\b|\bwoning\b|\bappartement\b|\bappart\b|\bflat\b|\bhuur\b|\bloyer\b"
            r"|\bmaison\b|\bhouse\b|\bverhuiz\w*|\bd[ée]m[ée]nag\w*|\bhypotheek\b|\bmortgage\b"
            r"|\bmakelaar\b|\bfunda\b|\bimmobili\w*|\bbezichtiging\w*"
        ),
        "familie": (
            r"\bfamil(?:ie|le|y)\b|\bvader\b|\bmoeder\b|\bouders\b|\bp[eè]re\b|\bm[eè]re\b"
            r"|\bparents\b|\bbroer(?:tje)?\b|\bzus(?:je)?\b|\bfr[eè]re\b|\bs(?:oe|œ)ur\b"
            r"|\boma\b|\bopa\b|\bmamie\b|\bpapi\b|\bschoon(?:ouders|moeder|vader)\b"
            r"|\bbeaux-parents\b|\bbelle-m[eè]re\b|\bbeau-p[eè]re\b|\btante\b|\boom\b|\boncle\b"
        ),
    }

    # Les 6, cleaning (roos_06.6, checked on the chat 2026-11-12): WhatsApp's own lines as
    # they appear in this (Dutch) export: calls, media, edits, deletions, encryption notice.
    style_system: tuple[str, ...] = (
        r"\b(?:afbeelding|audio|video|sticker|GIF|document|contactkaart)\s+weggelaten\b",
        r"<Dit bericht is bewerkt>",
        r"(?:Gemiste\s+)?(?:spraak|video)-?oproep\..*",
        r"(?:Je hebt dit bericht verwijderd|Dit bericht is verwijderd)\.?",
        r"Berichten en oproepen worden end-to-end versleuteld\..*",
    )
    letter_run: str = r"[^\W\d_]+"  # a run of letters: "c'est" -> "c", "est"
    emoji: str = "[\U0001f000-\U0001faff\U0001f1e6-\U0001f1ff☀-➿⬀-⯿️‍⃣]"
    # A token to mask: letters, possibly joined by hyphens ("est-ce", "peut-être").
    # Apostrophes, punctuation, digits ("0") and spaces stay as typed: the space before
    # "?" is a habit.
    masked_word: str = r"[^\W\d_]+(?:-[^\W\d_]+)*"

    @property
    def home_themes(self) -> dict[str, Labelled]:
        """Les 7's six themes in two groups. Work and holidays are left out: they hardly
        differ between the places. Sport reuses the climbing/running lists of lesson 3,
        with their known loose ends ("km", the climbing grades)."""
        # "(?i)" is only allowed at the start of a pattern, so it goes when joined
        sport = [self.topics[t].removeprefix("(?i)") for t in ("climbing", "running")]
        sport += [
            self.activities[t] for t in ("fietsen", "zwemmen", "yoga", "wandelen")
        ]
        going_out = ["uit eten", "feest & borrel"]
        return {
            "reizen": Labelled(Side.AWAY, self.theme_words["reizen"]),
            "klimmen & sport": Labelled(Side.AWAY, "|".join(sport)),
            "eten & uitgaan": Labelled(
                Side.AWAY,
                "|".join(
                    [self.activities[t] for t in going_out]
                    + [self.theme_words["eten & uitgaan"]]
                ),
            ),
            "Benji & kind": Labelled(Side.HOME, self.topics["child"]),
            "wonen": Labelled(Side.HOME, self.theme_words["wonen"]),
            "familie": Labelled(Side.HOME, self.theme_words["familie"]),
        }


# --- WordLists: hand-made lists of words and characters ------------------------------


class WordLists(Frozen):
    # Phones type ’ or ‘, people type ' : without this, the phone is the biggest "style"
    # difference.
    apostrophes: str = "’‘`´"
    # Short lists of the most frequent words per language, written out by hand (Roos chose
    # this over nltk, 2026-11-12). Words are matched after apostrophes are split off, so
    # "c'est" counts as "c" + "est". Only words in exactly one list count (see
    # `unique_stopwords`): "de", "je", "en", "la", "me", "que", "tu", "es" and "a" are in
    # two lists and so say nothing.
    stopwords: dict[str, frozenset[str]] = {
        "fr": frozenset(
            """
            le la les l un une des du de d au aux et ou où mais donc car ni je j tu il elle on
            nous vous ils elles me m te t se s lui leur moi toi ce c cette ces cet ça ca qui que
            qu quoi dont mon ma mes ton ta tes son sa ses notre nos votre vos dans pour avec sur
            sous sans chez par entre vers ne n pas plus rien jamais suis es est sommes êtes etes
            sont était etait étais etais été ete ai as a avons avez ont avais avait eu vais vas
            va allons allez vont fais fait faire faut peux peut veux veut dois doit très tres
            déjà deja aussi bien oui non ouais alors comme quand parce encore toujours beaucoup
            trop peu tout tous toute toutes cest jai voilà voila aujourd hui demain merci bisous
            coucou y en
            """.split()
        ),
        "en": frozenset(
            """
            the a an and or but is are was were be been am i you he she it we they me him her us
            them my your his its our their this that these those what which who whom to of in on
            at for with from by about as not no yes do does did have has had will would can
            could should just so if then than there here when where why how all some any very
            really too also im dont thats s t m ll re ve d
            """.split()
        ),
        "nl": frozenset(
            """
            de het een en of maar is zijn was waren ben bent ik jij je u hij zij ze wij we jullie
            mij me mijn jouw jou haar ons onze hun dit dat deze die wat wie welke naar van in op
            aan voor met uit door over als niet geen ja nee doen doet deed heb hebt heeft hadden
            had zal zou kan kunnen moet moeten wil willen gewoon ook nog al wel dan daar hier
            wanneer waar waarom hoe alle alles veel heel erg echt toch even nu s t
            """.split()
        ),
        "es": frozenset(
            """
            el la los las un una unos unas y o pero es son era fue ser estar está esta estoy yo
            tú tu él ella nosotros ellos me te se lo le les mi mis su sus nuestro este esto ese
            eso que qué quien de del en con por para sin sobre como cuando donde porque no sí si
            muy más mas también tambien ya hay hola gracias bueno pues todo nada algo
            """.split()
        ),
    }
    # ~150 French function words: they carry style, not topic. Accentless spellings
    # included, because that is how they are often typed.
    function_words: frozenset[str] = frozenset(
        """
        je j tu il elle on nous vous ils elles me m te t se s lui leur leurs y en moi toi soi
        eux ce c ceci cela ça ca celui celle ceux celles qui que qu quoi dont où ou lequel
        laquelle le la l les un une des du de d au aux mon ma mes ton ta tes son sa ses notre
        nos votre vos à a dans par pour sur sous avec sans chez entre vers contre depuis pendant
        avant après apres jusqu et mais donc or ni car si comme quand parce puisque
        être etre suis es est sommes êtes etes sont était etait étais etais été ete sera serait
        soit avoir ai as avons avez ont avait avais eu aura aurais aller vais vas va allons
        allez vont allé alle faire fais fait faisons faites font pouvoir peux peut pouvons
        pouvez peuvent pourrait pu vouloir veux veut voulons voulez veulent voudrais devoir
        dois doit devons devez doivent devrait falloir faut fallait faudra faudrait
        ne n pas plus jamais rien personne très tres trop beaucoup peu assez aussi bien encore
        déjà deja toujours oui non ouais voilà voila alors puis là ici est-ce peut-être
        peut-etre
        """.split()
    )
    mask: str = "x"  # what a content word becomes
    # PCA's sign is arbitrary: an axis whose marker trigrams point the wrong way is flipped.
    # x: correct French (low) to loose punctuation and English leftovers (high); y: school
    # French (low) to spoken French (high). Taken from the loadings in roos_06.6 §6.6.7.
    style_markers: dict[str, dict[str, tuple[str, ...]]] = {
        "x": {
            "high": (" ;)", " : ", " ! ", " or"),
            "low": ("doi", "veu", "hez", "peu"),
        },
        "y": {
            "high": (" ca", "ca ", "fau", "oui", " ? "),
            "low": ("bea", "uco", "-ce", " n'"),
        },
    }


# --- Limits: thresholds, what counts and what is left out ----------------------------


class Limits(Frozen):
    min_chars: int = (
        3  # shorter messages ("Ok", one emoji) are no evidence of a language
    )
    min_messages: int = 10  # Louis's messages in a week before it becomes a dot (les 5)
    min_per_group: int = (
        5  # French and other messages each, before a week gets a median gap
    )
    home_min_share: float = 0.08  # lines below this share are left out (les 7)
    french_share: float = (
        0.60  # of the recognised stopwords, for a message to be French
    )
    min_unit_words: int = 8  # a French unit shorter than this is dropped (les 6)
    min_fragment_words: int = 8  # a shorter remainder at the end of a unit is dropped
    min_month_fragments: int = 10  # fragments in a month before it counts in the trend
    min_arrow_share: float = 0.025  # of the x range: shorter arrows are skipped (les 6)


# --- Grouping: how messages are cut into windows, pieces and periods -----------------


class Grouping(Frozen):
    smooth_weeks: int = (
        4  # trailing rolling average in "Baby is taking over", as in 03.3
    )
    home_window: int = 3  # a "gesprekje": the message with the place and the 3 after it
    unit_gap_seconds: float = 5 * 60  # a person's French messages closer = one unit
    fragment_words: int = 15
    docs_per_quarter: int = 12
    fragments_per_doc: int = 40  # ~600 words
    # Louis wrote too little French in these quarters to sample from on their own.
    quarter_merge: dict[str, str] = {"2023Q3": "2023Q4", "2025Q2": "2025Q1"}
    # The chat ends in Sep 2026: the second half of 2026 is incomplete.
    half_year_merge: dict[str, str] = {"2026H2": "2026H1"}


# --- Tests: one class per shuffle test, all sharing the shuffle's own settings -------


class ShuffleTest(Frozen):
    n_shuffle: int = 10_000
    seed: int = 7


class TopicBalanceTest(ShuffleTest):
    """Les 3: child minus climbing, plus the time-order check of 04.6d."""

    run_weeks: int = 10  # runs of this many consecutive weeks


class HomeTest(ShuffleTest):
    """Les 7: Nieuwegein against the three cities of the long-distance years, taken
    together (Roos's sentence, 2026-11-12)."""

    place: str = "Nieuwegein"
    against: tuple[str, ...] = ("Parijs", "Amsterdam", "Utrecht")


class Tests(Frozen):
    topic_balance: TopicBalanceTest = TopicBalanceTest()
    symptoms: ShuffleTest = ShuffleTest()
    home: HomeTest = HomeTest()
    style: ShuffleTest = ShuffleTest()

    def with_n_shuffle(self, n: int) -> Tests:
        """The same tests with `n` shuffles each: the unit tests use 200, not 10,000."""
        return self.model_copy(
            update={
                name: test.model_copy(update={"n_shuffle": n}) for name, test in self
            }
        )


# --- Colors: grey-first (02.2), Okabe-Ito where a colour carries meaning -------------


class Palette(Frozen):
    grey: str = "#7f7f7f"
    light_grey: str = "#C8C8C8"
    dark_grey: str = "#333333"  # a fitted line over coloured dots


class BarsPalette(Palette):
    """Les 2."""

    before: str = "#cccccc"
    after: str = "#D55E00"  # Okabe-Ito vermillion


class TrendPalette(Palette):
    """Les 3."""

    child: str = "#0072B2"  # Okabe-Ito blue
    climbing: str = "#D55E00"  # Okabe-Ito vermillion


class PlanPalette(Palette):
    """Les 4: normal weeks grey, hard weeks crimson."""

    hard: str = "crimson"  # goad's PlotSettings().highlight_color


class FamiliarPalette(Palette):
    """Les 5: Okabe-Ito orange dots (Roos's pick, 2026-10-09), the line dark grey."""

    dots: str = "#E69F00"


class PaperPalette(Palette):
    """The two figures on off-white paper."""

    paper: str = "#fcfcfb"


class StylePalette(PaperPalette):
    """Les 6."""

    roos: str = "#2a78d6"
    louis: str = "#eb6834"
    text: str = "#1f1f1e"
    muted: str = "#8a8a83"
    frame: str = "#e6e6e1"

    @property
    def persons(self) -> dict[str, str]:
        return {"Roos": self.roos, "Louis": self.louis}


class HomePalette(PaperPalette):
    """Les 7."""

    away: str = "#eb6834"
    home: str = "#1baf7a"

    @property
    def sides(self) -> dict[str, str]:
        return {Side.AWAY: self.away, Side.HOME: self.home}


class Colors(Frozen):
    bars: BarsPalette = BarsPalette()
    trend: TrendPalette = TrendPalette()
    plan: PlanPalette = PlanPalette()
    familiar: FamiliarPalette = FamiliarPalette()
    style: StylePalette = StylePalette()
    home: HomePalette = HomePalette()


# --- the settings themselves: one of each --------------------------------------------

TIMELINE = Timeline()
CATEGORIES = Categories()
PATTERNS = Patterns()
WORD_LISTS = WordLists()
LIMITS = Limits()
GROUPING = Grouping()
TESTS = Tests()
COLORS = Colors()


# --- Paths: where each stage reads and writes ----------------------------------------


class Paths(Frozen):
    """Every stage boundary is a file here. `root` can be changed, so a test can point
    the whole pipeline at a temporary folder (`Paths(root=tmp_path)`)."""

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

    @property
    def weekly_french(self) -> Path:
        return self.processed / "3_weekly_french.parquet"

    @property
    def french_stats(self) -> Path:
        return self.processed / "3_french_stats.json"

    # The les-6 and les-7 figures start from the messages split back apart
    # (UnmergeMessages), a second stage-1 file, so les 2-5 keep the rows they were made with.
    @property
    def unmerged(self) -> Path:
        return self.processed / "1_unmerged.parquet"

    @property
    def unmerged_features(self) -> Path:
        return self.processed / "2_unmerged_features.parquet"

    @property
    def home_shares(self) -> Path:
        return self.processed / "3_home_shares.parquet"

    @property
    def home_stats(self) -> Path:
        return self.processed / "3_home_stats.json"

    @property
    def style_docs(self) -> Path:
        return self.processed / "3_style_docs.parquet"

    @property
    def style_means(self) -> Path:
        return self.processed / "3_style_means.parquet"

    @property
    def style_stats(self) -> Path:
        return self.processed / "3_style_stats.json"

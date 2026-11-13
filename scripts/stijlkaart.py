"""Roos and Louis in French: a style map from masked character trigrams (roos_06.6, §6.6.7).

Temporary home: Roos's code normally lives in `roos_pipeline/`; this file moves there once
she has approved the figure (her call, 2026-11-12). Until then its constants live here too,
so the move is one file.

The steps, in order:

    1_messages.parquet
      -> UnmergeMessages + CanonicalAuthor     one row per message, one name per person
      -> StripWhatsAppNoise                     system text and URLs out, apostrophes equal
      -> FrenchByStopwords                      French = >= 60% of the recognised stopwords
      -> french_units()                         a person's French messages < 5 min apart,
                                                units of 8+ words
      -> preprocess() -> fragments()            lowercase, no emoji, digits -> 0; 15 words
      -> mask_content_words()                   every non-function word becomes "x"
      -> sample_documents()                     12 samples of 40 fragments per person per quarter
      -> fit_style_space()                      tf-idf on char_wb trigrams, PCA(2)
      -> orient_axes() / half_year_means()      axis direction, trajectory points
      -> monthly_trend(), quarterly_similarity()  Spearman + permutation test
      -> draw_stijlkaart()                      the figure
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from goad_toolkit.datatransforms import TransformBase
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from roos_pipeline import settings
from scripts.sessionize import merge_messages, sessionize

SEED = settings.SEED

# --- cleaning ----------------------------------------------------------------------

# WhatsApp's own lines, as they appear in this (Dutch) export. Checked on the chat
# (2026-11-12): calls ("Spraakoproep. Geen antwoord", "Gemiste spraakoproep. Tik om terug te
# bellen"), media ("afbeelding weggelaten"), edits, deletions and the encryption notice.
SYSTEM_PATTERNS: tuple[str, ...] = (
    r"\b(?:afbeelding|audio|video|sticker|GIF|document|contactkaart)\s+weggelaten\b",
    r"<Dit bericht is bewerkt>",
    r"(?:Gemiste\s+)?(?:spraak|video)-?oproep\..*",
    r"(?:Je hebt dit bericht verwijderd|Dit bericht is verwijderd)\.?",
    r"Berichten en oproepen worden end-to-end versleuteld\..*",
)
# Phones type ’ or ‘, people type ' : without this, the phone is the biggest "style" difference.
APOSTROPHES = "’‘`´"

# --- French by stopwords -------------------------------------------------------------

# Short lists of the most frequent words per language, written out by hand (Roos chose this
# over nltk, 2026-11-12). Words are matched after apostrophes are split off, so "c'est"
# counts as "c" + "est". Only words in exactly one list count (see `unique_stopwords`):
# "de", "je", "en", "la", "me", "que", "tu", "es" and "a" are in two lists and so say nothing.
STOPWORDS: dict[str, frozenset[str]] = {
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
FRENCH_SHARE = 0.60  # of the recognised stopwords
LETTER_RUN = r"[^\W\d_]+"  # a run of letters: "c'est" -> "c", "est"

# --- units, fragments, masking --------------------------------------------------------

UNIT_GAP_SECONDS = 5 * 60  # a person's French messages closer than this form one unit
MIN_UNIT_WORDS = 8
FRAGMENT_WORDS = 15
MIN_FRAGMENT_WORDS = 8  # a shorter remainder at the end of a unit is dropped
EMOJI_PATTERN = "[\U0001f000-\U0001faff\U0001f1e6-\U0001f1ff☀-➿⬀-⯿️‍⃣]"
# A token to mask: letters, possibly joined by hyphens ("est-ce", "peut-être"). Apostrophes,
# punctuation, digits ("0") and spaces stay as typed: the space before "?" is a habit.
MASKED_WORD = r"[^\W\d_]+(?:-[^\W\d_]+)*"
MASK = "x"
# ~150 French function words: they carry style, not topic. Accentless spellings included,
# because that is how they are often typed.
FUNCTION_WORDS: frozenset[str] = frozenset(
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

# --- quarters, documents, half-years ------------------------------------------------------

# Louis wrote too little French in these quarters to sample from on their own.
QUARTER_MERGE = {"2023Q3": "2023Q4", "2025Q2": "2025Q1"}
# The chat ends in Sep 2026: the second half of 2026 is incomplete.
HALF_YEAR_MERGE = {"2026H2": "2026H1"}
DOCS_PER_QUARTER = 12
FRAGMENTS_PER_DOC = 40  # ~600 words
MIN_MONTH_FRAGMENTS = 10
N_PERMUTATIONS = 10_000

# --- the figure ---------------------------------------------------------------------------

ROOS_BLUE = "#2a78d6"
LOUIS_ORANGE = "#eb6834"
PERSON_COLORS = {"Roos": ROOS_BLUE, "Louis": LOUIS_ORANGE}
BACKGROUND = "#fcfcfb"
TEXT = "#1f1f1e"
MUTED = "#8a8a83"
FRAME = "#e6e6e1"
MIN_ARROW_SHARE = 0.025  # of the x range: shorter arrows are skipped


# === cleaning ===============================================================================


class StripWhatsAppNoise(TransformBase):
    """Remove WhatsApp's system text and URLs, make every apostrophe ', drop what is left empty."""

    def transform(self, data: pd.DataFrame, column: str = "message") -> pd.DataFrame:
        text = data[column].fillna("").str.replace("‎", "", regex=False)
        for pattern in (*SYSTEM_PATTERNS, settings.LINK_PATTERN):
            text = text.str.replace(pattern, " ", regex=True, flags=re.IGNORECASE)
        text = text.str.translate(str.maketrans(APOSTROPHES, "'" * len(APOSTROPHES)))
        data[column] = text.str.replace(r"\s+", " ", regex=True).str.strip()
        return data[data[column] != ""]


# === French by stopwords ===================================================================


def unique_stopwords(
    stopwords: dict[str, frozenset[str]] = STOPWORDS,
) -> dict[str, frozenset[str]]:
    """Per language, only the words that are in no other language's list."""
    return {
        language: words
        - frozenset().union(
            *(other for name, other in stopwords.items() if name != language)
        )
        for language, words in stopwords.items()
    }


def french_share(text: object, unique: dict[str, frozenset[str]]) -> float:
    """Share of the recognised (unique) stopwords that are French; NaN when none is recognised
    or when `text` is missing."""
    if not isinstance(text, str):
        return float("nan")
    counts = dict.fromkeys(unique, 0)
    for word in re.findall(LETTER_RUN, text.lower()):
        for language, words in unique.items():
            if word in words:
                counts[language] += 1
    recognised = sum(counts.values())
    return counts["fr"] / recognised if recognised else float("nan")


class FrenchByStopwords(TransformBase):
    """`is_french`: at least `threshold` of the recognised stopwords are French."""

    def transform(
        self,
        data: pd.DataFrame,
        column: str = "message",
        threshold: float = FRENCH_SHARE,
    ) -> pd.DataFrame:
        unique = unique_stopwords()
        data["french_share"] = data[column].map(lambda text: french_share(text, unique))
        data["is_french"] = data["french_share"] >= threshold
        return data


# === units, fragments, masking ============================================================


def french_units(
    messages: pd.DataFrame,
    gap: float = UNIT_GAP_SECONDS,
    min_words: int = MIN_UNIT_WORDS,
) -> pd.DataFrame:
    """A person's consecutive French messages less than `gap` seconds apart, joined into one
    unit; units under `min_words` words are dropped. The unit takes its first message's time."""
    french = messages[messages["is_french"]].copy()
    french["session_id"] = sessionize(french, "ts_local", "author", gap)
    units = merge_messages(french.sort_values("ts_local"), ["author", "session_id"])
    start = (
        french.groupby(["author", "session_id"])["ts_local"].min().rename("ts_local")
    )
    units = units.join(start, on=["author", "session_id"])
    units["n_words"] = units["message"].str.count(r"\S+")
    return units[units["n_words"] >= min_words].reset_index(drop=True)


def preprocess(text: str) -> str:
    """Lowercase, emoji out, every digit a 0, whitespace collapsed."""
    text = re.sub(EMOJI_PATTERN, " ", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"\d", "0", text)).strip()


def fragments(
    units: pd.DataFrame, size: int = FRAGMENT_WORDS, min_words: int = MIN_FRAGMENT_WORDS
) -> pd.DataFrame:
    """Cut every unit into pieces of `size` words (whitespace tokens); a remainder under
    `min_words` is dropped. Keeps `author` and `ts_local` of the unit."""
    rows = []
    for author, when, text in zip(
        units["author"], units["ts_local"], units["message"], strict=True
    ):
        words = preprocess(text).split()
        for start in range(0, len(words), size):
            piece = words[start : start + size]
            if len(piece) >= min_words:
                rows.append((author, when, " ".join(piece)))
    return pd.DataFrame(rows, columns=["author", "ts_local", "text"])


def mask_content_words(text: str, keep: frozenset[str] = FUNCTION_WORDS) -> str:
    """Every letter token that is not a function word becomes "x"; the rest stays as typed."""
    return re.sub(
        MASKED_WORD, lambda m: m.group(0) if m.group(0) in keep else MASK, text
    )


# === quarters, documents ==================================================================


def style_quarter(when: pd.Series) -> pd.Series:
    """ "2024Q1"-style labels, with the thin quarters merged (QUARTER_MERGE)."""
    return when.dt.to_period("Q").astype(str).replace(QUARTER_MERGE)


def sample_documents(
    frags: pd.DataFrame,
    n_docs: int = DOCS_PER_QUARTER,
    per_doc: int = FRAGMENTS_PER_DOC,
    seed: int = SEED,
) -> pd.DataFrame:
    """Per person per quarter, `n_docs` documents of `per_doc` random fragments each (with
    replacement when the quarter has fewer than `per_doc`). A document takes the median time
    of its fragments."""
    rng = np.random.default_rng(seed)
    rows = []
    for (author, quarter), group in frags.groupby(["author", "quarter"], sort=True):
        replace = len(group) < per_doc
        for _ in range(n_docs):
            pick = group.iloc[rng.choice(len(group), size=per_doc, replace=replace)]
            rows.append(
                (author, quarter, pick["ts_local"].median(), " ".join(pick["masked"]))
            )
    return pd.DataFrame(rows, columns=["author", "quarter", "ts_local", "text"])


# === the style space ======================================================================


def fit_style_space(
    docs: pd.DataFrame, seed: int = SEED
) -> tuple[TfidfVectorizer, PCA, np.ndarray]:
    """tf-idf of character trigrams within words (char_wb), then PCA to two components."""
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 3), min_df=5, sublinear_tf=True
    )
    weights = vectorizer.fit_transform(docs["text"]).toarray()
    pca = PCA(n_components=2, random_state=seed).fit(weights)
    return vectorizer, pca, pca.transform(weights)


def axis_loadings(
    vectorizer: TfidfVectorizer, pca: PCA, n: int = 12
) -> dict[str, dict[str, list[str]]]:
    """Per axis, the `n` trigrams with the lowest and the highest loading."""
    names = vectorizer.get_feature_names_out()
    out = {}
    for axis, component in zip(("x", "y"), pca.components_, strict=True):
        order = np.argsort(component)
        out[axis] = {
            "low": [names[i] for i in order[:n]],
            "high": [names[i] for i in order[::-1][:n]],
        }
    return out


def marker_score(
    vectorizer: TfidfVectorizer,
    component: np.ndarray,
    high: tuple[str, ...],
    low: tuple[str, ...],
) -> float:
    """Mean loading of the `high` marker trigrams minus that of the `low` ones (missing ones skipped).
    Positive: the axis already points the way the labels say."""
    index = {name: i for i, name in enumerate(vectorizer.get_feature_names_out())}

    def mean(markers: tuple[str, ...]) -> float:
        found = [component[index[m]] for m in markers if m in index]
        return float(np.mean(found)) if found else 0.0

    return mean(high) - mean(low)


def orient_axes(coords: np.ndarray, signs: tuple[int, int]) -> np.ndarray:
    """Multiply each axis by its sign (+1 or -1), so the picture agrees with its labels."""
    return coords * np.asarray(signs)


def half_year(when: pd.Series) -> pd.Series:
    """ "2024H1"-style labels, with the incomplete last half-year merged (HALF_YEAR_MERGE)."""
    half = np.where(when.dt.month <= 6, "H1", "H2")
    return (when.dt.year.astype(str) + half).replace(HALF_YEAR_MERGE)


def half_year_means(docs: pd.DataFrame, coords: np.ndarray) -> pd.DataFrame:
    """Per person per half-year, the mean position of the documents, in time order."""
    frame = docs[["author", "ts_local"]].assign(
        x=coords[:, 0], y=coords[:, 1], half=half_year(docs["ts_local"])
    )
    return (
        frame.groupby(["author", "half"], sort=True)
        .agg(x=("x", "mean"), y=("y", "mean"), ts_local=("ts_local", "median"))
        .reset_index()
    )


# === tests ===============================================================================


def permutation_spearman(
    order: np.ndarray, values: np.ndarray, n: int = N_PERMUTATIONS, seed: int = SEED
) -> tuple[float, float]:
    """Spearman's rho between `order` and `values`, and a two-sided p from shuffling `order`
    `n` times: (shuffles with |rho| at least as large + 1) / (n + 1)."""
    rng = np.random.default_rng(seed)
    x = rankdata(order)
    y = rankdata(values)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    rho = float(np.mean(x * y))
    shuffled = np.array([rng.permutation(x) for _ in range(n)])
    null = (shuffled * y).mean(axis=1)
    return rho, float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n + 1))


def monthly_trend(
    frags: pd.DataFrame,
    positions: np.ndarray,
    author: str,
    axis: int,
    min_fragments: int = MIN_MONTH_FRAGMENTS,
) -> dict:
    """Per month the mean position of one person's fragments on `axis` (months with at least
    `min_fragments`), and the permutation Spearman between month and position."""
    mine = frags["author"].to_numpy() == author
    month = frags.loc[mine, "ts_local"].dt.to_period("M")
    per_month = pd.DataFrame(
        {"month": month.to_numpy(), "pos": positions[mine, axis]}
    ).groupby("month")["pos"]
    means = per_month.mean()[per_month.size() >= min_fragments]
    rho, p = permutation_spearman(np.arange(len(means)), means.to_numpy())
    return {"rho": rho, "p": p, "n_months": len(means), "means": means}


def quarterly_similarity(docs: pd.DataFrame, vectorizer: TfidfVectorizer) -> pd.Series:
    """Per quarter, the cosine between Roos's and Louis's mean tf-idf profile."""
    weights = vectorizer.transform(docs["text"]).toarray()
    out = {}
    for quarter in sorted(docs["quarter"].unique()):
        mean = {
            a: weights[
                ((docs["quarter"] == quarter) & (docs["author"] == a)).to_numpy()
            ].mean(axis=0)
            for a in ("Roos", "Louis")
        }
        out[quarter] = float(
            mean["Roos"]
            @ mean["Louis"]
            / (np.linalg.norm(mean["Roos"]) * np.linalg.norm(mean["Louis"]))
        )
    return pd.Series(out, name="cosine")


# === the figure ===========================================================================


def shade(color: str, t: float) -> tuple[float, float, float]:
    """The person's colour from light (t = 0, 2023) to full (t = 1, 2026)."""
    weight = 0.25 + 0.75 * float(np.clip(t, 0, 1))
    r, g, b = to_rgb(color)
    return (1 - weight * (1 - r), 1 - weight * (1 - g), 1 - weight * (1 - b))


def p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def draw_stijlkaart(
    docs: pd.DataFrame, coords: np.ndarray, means: pd.DataFrame, texts: dict
) -> Figure:
    """The style map. `texts` holds every sentence on it (title, subtitle, the axis words and
    their examples, footnote), so the wording is decided in the notebook, not here."""
    start, end = docs["ts_local"].min(), docs["ts_local"].max()

    def time_share(when: pd.Series) -> np.ndarray:
        return ((when - start) / (end - start)).to_numpy(dtype=float)

    fig = plt.figure(figsize=(12, 10), dpi=150, facecolor=BACKGROUND)
    ax = fig.add_axes((0.1, 0.2, 0.85, 0.62), facecolor=BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color(FRAME)
    ax.set_xticks([])
    ax.set_yticks([])

    for author, color in PERSON_COLORS.items():
        mine = (docs["author"] == author).to_numpy()
        ax.scatter(
            coords[mine, 0],
            coords[mine, 1],
            s=16,
            alpha=0.35,
            linewidths=0,
            color=[shade(color, t) for t in time_share(docs.loc[mine, "ts_local"])],
        )

    x_range = np.ptp(coords[:, 0])
    for author, color in PERSON_COLORS.items():
        path = (
            means[means["author"] == author]
            .sort_values("ts_local")
            .reset_index(drop=True)
        )
        t = time_share(path["ts_local"])
        points = path[["x", "y"]].to_numpy()
        for i in range(1, len(points)):
            if np.linalg.norm(points[i] - points[i - 1]) < MIN_ARROW_SHARE * x_range:
                continue
            ax.add_patch(
                FancyArrowPatch(
                    points[i - 1],
                    points[i],
                    arrowstyle="-|>",
                    mutation_scale=18,
                    linewidth=3,
                    color=shade(color, t[i]),
                    shrinkA=6,
                    shrinkB=6,
                    zorder=3,
                )
            )
        sizes = np.full(len(points), 40.0)
        sizes[[0, -1]] = 110.0
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=sizes,
            color=[shade(color, ti) for ti in t],
            zorder=4,
            linewidths=0,
        )
        for idx, neighbour, year in ((0, 1, start.year), (-1, -2, end.year)):
            # away from the trajectory, measured in axis widths so x and y count the same
            away = (points[idx] - points[neighbour]) / np.ptp(coords, axis=0)
            away = away / (np.linalg.norm(away) + 1e-12)
            ax.annotate(
                f"{author} {year}",
                points[idx],
                xytext=(34 * away[0], 34 * away[1]),
                textcoords="offset points",
                ha="left" if away[0] >= 0 else "right",
                va="bottom" if away[1] >= 0 else "top",
                fontsize=13,
                fontweight="bold",
                color=color,
                zorder=5,
                bbox={
                    "facecolor": BACKGROUND,
                    "edgecolor": "none",
                    "alpha": 0.85,
                    "pad": 1.5,
                },
            )

    ax.text(
        0.985,
        0.025,
        "grote stip = gemiddelde per halfjaar · kleine stippen = steekproeven van "
        "±600 woorden\nlicht = 2023, donker = 2026",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=MUTED,
    )

    # axis ends: a bold word, grey examples underneath
    ax.text(
        0,
        -0.035,
        texts["x_low"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        0,
        -0.085,
        texts["x_low_examples"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color=MUTED,
    )
    ax.text(
        1,
        -0.035,
        texts["x_high"],
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        1,
        -0.085,
        texts["x_high_examples"],
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color=MUTED,
    )
    ax.text(
        -0.06,
        0,
        texts["y_low"],
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        -0.03,
        0,
        texts["y_low_examples"],
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=10.5,
        color=MUTED,
    )
    ax.text(
        -0.06,
        1,
        texts["y_high"],
        transform=ax.transAxes,
        ha="center",
        va="top",
        rotation=90,
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        -0.03,
        1,
        texts["y_high_examples"],
        transform=ax.transAxes,
        ha="center",
        va="top",
        rotation=90,
        fontsize=10.5,
        color=MUTED,
    )

    fig.text(
        0.1,
        0.955,
        texts["title"],
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.1,
        0.895,
        texts["subtitle"],
        ha="left",
        va="top",
        fontsize=13,
        color=TEXT,
        linespacing=1.6,
        alpha=0.8,
    )
    fig.text(
        0.1,
        0.105,
        texts["footnote"],
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED,
        linespacing=1.6,
    )
    return fig

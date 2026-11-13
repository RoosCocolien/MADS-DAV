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
# A "word" for the group keywords (6.5): 3+ letters, no digits, so "je", "ok" and "18:51" never count.
KEYWORD_PATTERN = r"(?u)\b[^\W\d_]{3,}\b"

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

# --- lesson 5: Louis's French and message length (roos_05.6, 2026-10-09) ------------

# Words are runs of characters without a space: an emoji or a link is one word.
WORD_PATTERN = r"\S+"
MIN_MESSAGES = 10  # Louis's messages in a week before it becomes a dot (fewer: noisy)
MIN_PER_GROUP = 5  # French and other messages each, before a week gets a median gap

# --- lesson 6: Louis in French vs. English (roos_06.5, 2026-10-16) -----------------

COMPARE_LANGUAGES = ["fr", "en"]
# Multilingual, so "je t'aime" and "I love you" land on (almost) the same vector; the
# 6.4 model (all-MiniLM-L6-v2) is English-only. 384 dimensions, fixed by the model.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_UNIT_WORDS = 4  # shorter units are mostly "hahaha", "merci", "bonne nuit" (6.4)
LENGTH_BANDS = (8, 20, 40)  # words: 4-8, 9-20, 21-40, 41+ — held fixed by the shuffle
MIN_UNITS_PER_WEEK = 5  # per language, before a week's share is drawn
# Blue against vermillion: the Okabe-Ito pair furthest apart, also for colour-blind eyes. The
# purple/green pair before (2026-10-26) ran together where thousands of small dots overlap.
FRENCH_COLOR = "#0072B2"  # Okabe-Ito blue
ENGLISH_COLOR = "#D55E00"  # Okabe-Ito vermillion

# --- 6.5.10: topics from word lists, apart from the encoder (teacher's feedback, 2026-10-30) --
# Run on normalised text (`features.normalise`: lowercase, no accents, straight apostrophes),
# so "chérie", "cherie" and "Chérie" are one term. One regex per term, so the notebook can
# count how often each term fires. Generic words were checked on the units and left out:
# "love" ("I love sauna"), "cute" ("he's very cute", a dog), "beautiful" ("Paris is
# beautiful"), "avec toi"/"with you", "bonne nuit"/"good night" (a routine sign-off, the
# loving ones carry 😘 anyway), "tomorrow"/"demain" (in 1 of 13 English units), French
# "car" (= "because"), "book" (mostly a book being read).
LOVE_TERMS: tuple[str, ...] = (
    # Roos's list (2026-10-30), French
    r"\bmon cheri|\bma cherie",
    r"\bmon amour\b",
    r"\bje t'?aime\b|\bt'aime\b",
    r"\bje pense (?:tres fort |fort |beaucoup )?a toi\b",
    r"\bcourage\b",
    r"\bje crois en toi\b",
    r"\bkiff",  # also "tu as kiffé ?" = did you enjoy it; see the term counts
    # ... and in English
    r"\bmy (?:love|dear)\b|\bdarling\b|\bhoney\b|\bsweetheart\b",
    r"\b(?:i )?love you\b",
    r"\bthinking (?:of|about) you\b|\bi think (?:a lot |strongly )?(?:of|about) you\b",
    r"\bbelieve in you\b",
    # found in Louis's units (6.5.10), French
    r"\btu me manques\b|\bmanques? (?:a moi|aussi)\b",
    r"\bhate de te (?:voir|retrouver|serrer)\b",
    r"\bfais de beaux reves\b|\bdors bien\b",
    r"\bma belle\b|\bmon coeur\b|\bla plus belle\b|\btu es (?:deja |si |tres |trop )?belle\b",
    r"\b(?:t'es|tu es|toi) la meilleure\b",
    r"\btrop forte\b|\bfier de toi\b|\btu geres\b|\bbravo\b|\bbien joue\b|\bbonne chance\b",
    r"\bbisous?\b|\bcalins?\b|\bje t'embrasse\b|\bfaire l'amour\b|\bbeaucoup d'amour\b|\btrop mignonne\b",
    # ... and in English
    r"\bmiss you\b",
    r"\bcan't wait to see you\b|\blooking forward to (?:see|seeing|hug|hugging|kiss) you\b",
    r"\bsweet dreams\b|\bsleep well\b",
    r"\byou(?:'re| are) (?:so |very )?(?:beautiful|the best|amazing|strong)\b",
    r"\bproud of you\b|\byou can do it\b|\bwell done\b|\bgood luck\b",
    r"\bhugs?\b|\bhugging\b|\bkiss(?:es|ing)?\b|\bmake love\b|\bmaking love\b",
    # emoji: hearts and kisses, and the encouraging ones (💪 is mostly "good luck 💪")
    r"[😘🫶💛🧡❤♥😍🥰💕💖💗💪👏🍀]",
)
PLANNING_TERMS: tuple[str, ...] = (
    # travel and times
    r"\btrains?\b|\bgare\b|\bstation\b|\beurostar\b|\bthalys\b|\btgv\b|\bouigo\b|\bsncf\b",
    r"\bbillets?\b|\btickets?\b|\bhoraires?\b|\bschedule\b|\bretard\b|\bdelay(?:ed)?\b",
    r"\bdepart(?:ure)?\b|\bleave at\b|\bje pars\b|\barriv(?:e|al|ing|er)\b|\bj'arrive\b",
    r"\bwhat time\b|\bquelle heure\b|\b\d{1,2}[:h]\d{2}\b|\b\d{1,2} ?(?:am|pm)\b",
    r"\bflights?\b|\bplane\b|\bavion\b|\bair?port\b|\baeroport\b",
    r"\bbus\b|\bmetro\b|\btram\b|\bvoiture\b|\b(?:by|the|my|a|your|his) car\b|\bdriv(?:e|ing)\b|\bconduire\b|\bparking\b|\bvelo\b|\bbike\b",
    # cities he travels between (Paris is also where he lives: "it's raining in Paris" counts too)
    r"\bparis\b|\butrecht\b|\bamsterdam\b|\brotterdam\b|\blyon\b|\bmarseille\b|\bbordeaux\b|\blille\b",
    r"\bnantes\b|\btoulouse\b|\bgrenoble\b|\bbru(?:ssels|xelles)\b|\blond(?:on|res)\b|\bbarcelon[ae]\b|\bauxerre\b|\bbressuire\b",
    # house, moving, cleaning, interior
    r"\bmaison\b|\bhouse\b|\bappart(?:ement)?\b|\bapp?art?ment\b|\bflat\b|\bloyer\b|\brent\b",
    r"\bdemenag|\bmoving\b|\bmove in\b",
    r"\bnettoy|\bmenage\b|\bclean(?:ing|ed)?\b|\baspirateur\b|\bvacuum\b|\blessive\b|\blaundry\b|\bvaisselle\b|\bdishes\b",
    r"\bmeubles?\b|\bfurniture\b|\bikea\b|\bcanape\b|\bsofa\b|\bcouch\b|\bcuisine\b|\bkitchen\b",
    r"\brideaux?\b|\bcurtains?\b|\bpeinture\b|\bpaint(?:ing)?\b|\bdeco(?:ration)?\b|\binterieur\b|\binterior\b",
    r"\bleboncoin\b|\bmarktplaats\b|\bgroceries\b|\bsupermarche\b",
    # appointments and plans
    r"\brendez-vous\b|\brdv\b|\bappointment\b|\breserv(?:ation|er|e|ed)?\b|\bbooking\b|\bbooked\b|\bto book\b",
    r"\bplanning\b|\bplanned\b|\bthis (?:weekend|week end)\b|\bce week-?end\b",
)
UNIT_TOPIC_TERMS: dict[str, tuple[str, ...]] = {
    "love": LOVE_TERMS,
    "planning": PLANNING_TERMS,
}
# Okabe-Ito, away from the language pair (blue/vermillion); "other" stays grey (02.2).
TOPIC_COLORS = {
    "love": "#CC79A7",
    "planning": "#009E73",
    "both": "#E69F00",
    "other": "#C8C8C8",
}  # LIGHT_GREY

# --- lesson 7: places that come up on the same day (roos_07.4, 2026-10-30) ---------

# The long-distance period runs from the first message (12 Aug 2023) to the move.
LONG_DISTANCE_START = pd.Timestamp("2023-08-12")
# Every place that came up in the chat, found by listing its capitalised words
# (2026-10-30) and checking the ambiguous ones by hand. Matched case-insensitively on the
# raw text, not on `normalise`d text: without accents "corsé" (strong) becomes "corse".
# Left out on purpose: "Nice" (793 messages, nearly all the English word), "Tours" ("deux
# tours"), "Spa" (also "au spa de Maarssen"), "Houten" ("houten deur"), Luxembourg (the
# country), Schiphol (an airport), and neighbourhoods and streets (Montmartre, Bastille,
# Overvecht, Sloterdijk, ...), which belong to a city already on the list.
PLACES: dict[str, tuple[str, str]] = {
    # France
    "Paris": ("France", r"\bparis\b|\bparijs\b"),
    "Marseille": ("France", r"\bmarseille\b"),
    "Lyon": ("France", r"\blyon\b"),
    "Lille": ("France", r"\blille\b"),
    "Avignon": ("France", r"\bavignon\b"),
    "Toulouse": ("France", r"\btoulouse\b"),
    "Bressuire": ("France", r"\bbressuire\b"),
    "Villejuif": ("France", r"\bvillejuif\b"),
    "Toulon": ("France", r"\btoulon\b"),
    "Bordeaux": ("France", r"\bbordeaux\b"),
    "Levallois": ("France", r"\blevallois\b"),
    "Cholet": ("France", r"\bcholet\b"),
    "Nantes": ("France", r"\bnantes\b"),
    "Angers": ("France", r"\bangers\b"),
    "Rungis": ("France", r"\brungis\b"),
    "Grenoble": ("France", r"\bgrenoble\b"),
    "Nancy": ("France", r"\bnancy\b"),
    "Metz": ("France", r"\bmetz\b"),
    "La Rochelle": ("France", r"\bla ?rochelle\b"),
    "Fontainebleau": ("France", r"\bfontainebleau\b"),
    "Montmorency": ("France", r"\bmontmorency\b"),
    "Briançon": ("France", r"\bbrian[cç]on\b"),
    "Strasbourg": ("France", r"\bstrasbourg\b"),
    "Courlay": ("France", r"\bcourlay\b"),
    "Tergnier": ("France", r"\btergnier\b"),
    "Brest": ("France", r"\bbrest\b"),
    "Poitiers": ("France", r"\bpoitiers\b"),
    "Dinan": ("France", r"\bdinan\b"),
    "Nevers": ("France", r"\bnevers\b"),
    "Maubeuge": ("France", r"\bmaubeuge\b"),
    "Margency": ("France", r"\bmargency\b"),
    "Maisons-Alfort": ("France", r"\balfort\b"),
    "Vincennes": ("France", r"\bvincennes\b"),
    "Auxerre": ("France", r"\bauxerre\b"),
    "Montpellier": ("France", r"\bmontpellier\b"),
    "Rennes": ("France", r"\brennes\b"),
    "Rouen": ("France", r"\brouen\b"),
    "Orléans": ("France", r"\borl[eé]ans\b"),
    "Annecy": ("France", r"\bannecy\b"),
    "Chamonix": ("France", r"\bchamonix\b"),
    "Versailles": ("France", r"\bversailles\b"),
    # Corsica, where we met: the island itself, and the towns on it
    "Corsica": ("France", r"\bcorse\b|\bcorsica\b"),
    "Corte": ("France", r"\bcorte\b"),
    "Piana": ("France", r"\bpiana\b"),
    "Calvi": ("France", r"\bcalvi\b"),
    "L'Île-Rousse": ("France", r"\b[iî]le[- ]rousse\b"),
    "Vizzavona": ("France", r"\bviz+[ae]vona\b"),
    # Netherlands
    "Utrecht": ("Netherlands", r"\butrecht?\b"),
    "Amsterdam": ("Netherlands", r"\bamsterdam(?:se)?\b"),
    "Rotterdam": ("Netherlands", r"\brotterdam\b"),
    "Zoetermeer": ("Netherlands", r"\bzoetermeer\b"),
    "Nieuwegein": ("Netherlands", r"\bnieuwegein\b"),
    "Groningen": ("Netherlands", r"\bgroningen\b"),
    "Terschelling": ("Netherlands", r"\bterschelling\b"),
    "Gouda": ("Netherlands", r"\bgouda\b"),
    "Den Haag": ("Netherlands", r"\bhaag\b|\bla haye\b|\bthe hague\b"),
    "Haarlem": ("Netherlands", r"\bhaarlem\b"),
    "IJsselstein": ("Netherlands", r"\bijsselstein\b"),
    "Eindhoven": ("Netherlands", r"\beindhoven\b"),
    "Maarssen": ("Netherlands", r"\bmaars+en\b"),
    "Almelo": ("Netherlands", r"\balmelo\b"),
    "Nijmegen": ("Netherlands", r"\bnijmegen\b"),
    "Amersfoort": ("Netherlands", r"\bamersfoort\b"),
    "Zeist": ("Netherlands", r"\bzeist\b"),
    "Leerdam": ("Netherlands", r"\bleerdam\b"),
    "Maastricht": ("Netherlands", r"\bmaastricht\b"),
    "Winschoten": ("Netherlands", r"\bwinschoten\b"),
    "Pekela": ("Netherlands", r"\bpekela\b"),
    "Opsterland": ("Netherlands", r"\bopsterland\b"),
    "Duivendrecht": ("Netherlands", r"\bduivendrecht\b"),
    "De Bilt": ("Netherlands", r"\bde bilt\b"),
    "Bilthoven": ("Netherlands", r"\bbilthoven\b"),
    "Wageningen": ("Netherlands", r"\bwageningen\b"),
    "Montfoort": ("Netherlands", r"\bmontfoort\b"),
    "Arnhem": ("Netherlands", r"\barnhem\b"),
    "Maartensdijk": ("Netherlands", r"\bmaartensdijk\b"),
    "Wilnis": ("Netherlands", r"\bwilnis\b"),
    "Vianen": ("Netherlands", r"\bvianen\b"),
    "Kinderdijk": ("Netherlands", r"\bkinderdijk\b"),
    "Lage Vuursche": ("Netherlands", r"\bvuursche\b"),
    "Delft": ("Netherlands", r"\bdelft\b"),
    "Leiden": ("Netherlands", r"\bleiden\b"),
    "Deventer": ("Netherlands", r"\bdeventer\b"),
    "Den Bosch": ("Netherlands", r"\bden bosch\b|\b's-hertogenbosch\b"),
    "Breda": ("Netherlands", r"\bbreda\b"),
    "Tilburg": ("Netherlands", r"\btilburg\b"),
    "Alkmaar": ("Netherlands", r"\balkmaar\b"),
    "Vlieland": ("Netherlands", r"\bvlieland\b"),
    "Woerden": ("Netherlands", r"\bwoerden\b"),
    "Baarn": ("Netherlands", r"\bbaarn\b"),
    "Soest": ("Netherlands", r"\bsoest\b"),
    "Culemborg": ("Netherlands", r"\bculemborg\b"),
    # Elsewhere
    "Brussels": ("Elsewhere", r"\bbrussels?\b|\bbruxelles\b"),
    "Barcelona": ("Elsewhere", r"\bbarcelon[ae]\b"),
    "Basel": ("Elsewhere", r"\bbasel\b"),
    "Dinant": ("Elsewhere", r"\bdinant\b"),
    "Leuven": ("Elsewhere", r"\bleuven\b|\blouvain\b"),
    "Gent": ("Elsewhere", r"\bgent\b|\bghent\b|\bgand\b"),
    "Antwerp": ("Elsewhere", r"\banvers\b|\bantwerpen\b|\bantwerp\b"),
    "Hotton": ("Elsewhere", r"\bhotton\b"),
    "Berdorf": ("Elsewhere", r"\bberdorf\b"),
    "Aachen": ("Elsewhere", r"\baken\b|\baachen\b|\baix[- ]la[- ]chapelle\b"),
    "Prague": ("Elsewhere", r"\bprague\b|\bpraag\b|\bpraha\b"),
    "Berlin": ("Elsewhere", r"\bberlin\b|\bberlijn\b"),
    "London": ("Elsewhere", r"\blondon\b|\blondres\b|\blonden\b"),
    "Rome": ("Elsewhere", r"\brome\b|\broma\b"),
    "Bologna": ("Elsewhere", r"\bbologna\b"),
    "Milan": ("Elsewhere", r"\bmilano\b"),
    "Valencia": ("Elsewhere", r"\bvalencia\b"),
    "Porto": ("Elsewhere", r"\bporto\b"),
    "Tenerife": ("Elsewhere", r"\btenerife\b"),
    "Mallorca": ("Elsewhere", r"\bmallorca\b"),
    "Tokyo": ("Elsewhere", r"\btokyo\b"),
    "Philadelphia": ("Elsewhere", r"\bphiladelphia\b"),
}
COUNTRY_ORDER = ["France", "Netherlands", "Elsewhere"]
# Okabe-Ito blue for France (as FRENCH_COLOR), orange for the Netherlands; the rest grey.
COUNTRY_COLORS = {
    "France": "#0072B2",
    "Netherlands": "#E69F00",
    "Elsewhere": "#7f7f7f",
}  # GREY
# The clutter-reduced graph (Roos's rules, 2026-10-30): a place needs MIN_PLACE_DAYS days,
# a link MIN_LINK_DAYS days together, and a link stays if it is among the MAX_LINKS
# strongest of one of its places. A place left alone keeps its strongest link (made less
# strict the same day: "top 3 of both places" left 11 of 16 places without a link).
MIN_PLACE_DAYS = 10
MIN_LINK_DAYS = 5
MAX_LINKS = 3
ALWAYS_KEEP = ("Corsica",)  # where we met: kept whatever its count
# Who names a place (Roos's rule, 2026-11-03): Roos's share of its days. At SHARED_HIGH or
# more it is Roos's place, at SHARED_LOW or less Louis's, in between it is shared.
SHARED_LOW = 0.40
SHARED_HIGH = 0.60
# Okabe-Ito reddish purple and bluish green: not the country colours, so the two
# pictures can't be confused; shared is grey.
ROOS_COLOR = "#CC79A7"
LOUIS_COLOR = "#009E73"
SHARED_COLOR = "#7f7f7f"  # GREY
# People near Benji, looking for babysitters (Roos's rules, 2026-11-03): a person is near
# Benji when named within BENJI_WINDOW of a message naming Benji, from the first "Benji"
# on. Counted in days. The names themselves live in config.toml [people], not here.
BENJI = "Benji"
BENJI_PATTERN = r"\bbenji"
BENJI_WINDOW = pd.Timedelta(minutes=30)
# days near Benji before a person gets a dot (a starting point, 2026-11-03)
MIN_NEAR_DAYS = 2
# Okabe-Ito vermillion for Benji, sky blue for a family word, grey for a name.
BENJI_COLOR = "#D55E00"
FAMILY_COLOR = "#56B4E9"
NAME_COLOR = "#7f7f7f"  # GREY
# Who connects our friends and family (Roos's rules, 2026-11-03): two people are linked
# when named within FRIEND_WINDOW of each other, or (second picture) in the same message;
# their side is who names them (SHARED_LOW/HIGH). The connector is the highest
# betweenness, unweighted as in 07.2, on the full graph; only the picture is pruned.
FRIEND_WINDOW = pd.Timedelta(minutes=30)
MIN_PERSON_DAYS = 10  # in the picture: a person named on at least this many days
MIN_PAIR_DAYS = 3  # in the picture: a link of at least this many days (starting points)
N_CONNECTORS = 3  # the top betweenness, ringed
# Topics we talk about (Roos asked for the 15 most discussed hobbies and activities,
# 2026-11-03). A CANDIDATE list, (group, regex) like PLACES, so `place_days` counts them
# in days. Matched after links are removed: a Spotify or Garmin URL is no film or music.
# Checked on samples (2026-11-03) and left out because they mostly meant something else:
# "dans" (French "in"), "les"/"cours" (French), "course" ("of course"), "soirée" (an
# evening), "photo" (sending one), "gym" (mostly the climbing gym), "boeken" (booking).
# Still loose: "vélo" is also transport, "km" also cycling, "prise" also "taken".
ACTIVITY_TOPICS: dict[str, tuple[str, str]] = {
    # sport
    "klimmen": ("sport", TOPIC_PATTERNS["climbing"]),
    "hardlopen": ("sport", TOPIC_PATTERNS["running"]),
    "fietsen": (
        "sport",
        r"\bfiets(?:en|tocht)?\b|\bv[ée]lo\b|\bcycling\b|\bbike ride\b|\bwielren",
    ),
    "zwemmen": (
        "sport",
        r"\bzwem(?:men|bad)?\b|\bnager\b|\bnatation\b|\bpiscine\b|\bswim(?:ming)?\b",
    ),
    "yoga": ("sport", r"\byoga\b"),
    "wandelen": (
        "sport",
        r"\bwandel(?:en|ing|tocht)?\b|\brandonn[ée]e\b|\bhik(?:e|ing)\b|\bbalade à pied\b|\bpromenade\b|\bwalk(?:ing)?\b",
    ),
    "skiën & bergen": (
        "sport",
        r"\bski(?:[ëe]n|ing)?\b|\bsnowboard\b|\bmontagne\b|\bmountains?\b",
    ),
    "fitness & balsport": (
        "sport",
        r"\bfitness\b|\bsportschool\b|\bmuscu(?:lation)?\b|\bpil[âa]tes\b|\bcrossfit\b|\bpadel\b|\btennis\b|\bvoetbal\b|\bfootball\b",
    ),
    "dansen": ("sport", r"\bdansen\b|\bdansles\b|\bdanser\b|\bdancing\b|\bsalsa\b"),
    "zeilen & water": (
        "sport",
        r"\bzeil(?:en|boot)\b|\bsail(?:ing)?\b|\bkayak\b|\bsurf(?:en|ing)?\b|\bkano\b|\bpaddle\b",
    ),
    # culture
    "film": (
        "cultuur",
        r"\bfilms?\b|\bmovies?\b|\bcin[ée]ma\b|\bbioscoop\b|\bnetflix\b",
    ),
    "muziek & concerten": (
        "cultuur",
        r"\bconcerts?\b|\bmusique\b|\bmuziek\b|\bmusic\b|\bfestival\b|\bguitare?\b|\bgitaar\b|\bpiano\b",
    ),
    "lezen & boeken": (
        "cultuur",
        r"\bboek\b|\bbooks?\b|\blivres?\b|\blezen\b|\blire\b|\breading\b|\broman\b|\bbouquiner\b",
    ),
    "museum & expo": (
        "cultuur",
        r"\bmus[ée]e\b|\bmuseum\b|\bexpo(?:sitie|sition)?s?\b|\bexhibition\b|\bgalerie\b|\bgallery\b",
    ),
    "theater": (
        "cultuur",
        r"\bth[ée][aâ]tre\b|\btheater\b|\bvoorstelling\b|\bspectacle\b|\bcabaret\b",
    ),
    "fotografie": (
        "cultuur",
        r"\bfotografe?(?:ie|ren)\b|\bphotograph(?:e|ie|y|er)\b|\bappareil photo\b",
    ),
    "spelletjes & schaken": (
        "cultuur",
        r"\bspelletjes?\b|\bjeux? de soci[ée]t[ée]\b|\bboard ?games?\b|\bjeux? vid[ée]o\b|\bpuzzel\b|\bpuzzle\b|\bschaken\b|\bchess\b|\blichess\b",
    ),
    # going out, at home
    "koken": (
        "uit & thuis",
        r"\bkoken\b|\bcuisiner\b|\bcooking\b|\bcook\b|\brecette\b|\brecept\b|\brecipe\b",
    ),
    "uit eten": (
        "uit & thuis",
        r"\brestaurant\b|\bresto\b|\buit eten\b|\bbrunch\b|\bterras(?:je)?\b",
    ),
    "feest & borrel": (
        "uit & thuis",
        r"\bf[eê]tes?\b|\bfeest(?:je)?\b|\bparty\b|\bborrel\b|\bap[ée]ro\b|\bdrinks\b",
    ),
    "sauna & massage": (
        "uit & thuis",
        r"\bsauna\b|\bmassages?\b|\bhammam\b|\bwellness\b",
    ),
    "shoppen": ("uit & thuis", r"\bshopp?(?:en|ing)\b|\bwinkelen\b|\bsoldes\b"),
    "tuin & planten": (
        "uit & thuis",
        r"\btuinieren\b|\bmoestuin\b|\bpotager\b|\bjardinage\b|\bgardening\b|\bplantes?\b|\bplanten\b",
    ),
    # life: not hobbies, but Roos chose to count them (2026-11-03)
    "Benji & ouderschap": ("leven", TOPIC_PATTERNS["child"]),
    "werk": (
        "leven",
        r"\bwerk\b|\btravail\b|\bboulot\b|\bwork(?:ing)?\b|\bjob\b|\bcoll[eè]gues?\b|\bcolleagues?\b|\bmeeting\b|\br[ée]union\b",
    ),
    "studie & school": (
        "leven",
        r"\bhogeschool\b|\bstudie\b|\b[ée]tudes\b|\bstudy\b|\bexam(?:en)?s?\b|\blessen\b|\btentamen\b|\bth[eè]se\b|\bscriptie\b|\bmaster\b",
    ),
    "vakantie & reizen": (
        "leven",
        r"\bvakantie\b|\bvacances\b|\bholiday\b|\bvacation\b|\bvoyage\b|\breis\b|\btrip\b|\bcamping\b",
    ),
}
N_TOPICS = 15  # the most discussed, in days
LINK_PATTERN = r"https?://\S+|www\.\S+"

# --- 7.4.13: from on the road to home (Roos's rules, 2026-11-12) -----------------------
# Five cities, Dutch names; Paris includes Gare du Nord, which is also a "reizen" word,
# so a Gare du Nord message always counts as travel too.
HOME_PLACES: dict[str, str] = {
    "Parijs": PLACES["Paris"][1] + r"|\bgare du nord\b",
    "Amsterdam": PLACES["Amsterdam"][1],
    "Utrecht": PLACES["Utrecht"][1],
    "Zoetermeer": PLACES["Zoetermeer"][1],
    "Nieuwegein": PLACES["Nieuwegein"][1],
}
# Six themes in two groups, (group, regex). Work and holidays are left out: they hardly
# differ between the places. Sport reuses the climbing/running lists of lesson 3, with
# their known loose ends ("km", the climbing grades).
AWAY, HOME = "onderweg", "thuis"
HOME_THEMES: dict[str, tuple[str, str]] = {
    "reizen": (
        AWAY,
        r"\btrein\b|\btrains?\b|\btgv\b|\bthalys\b|\beurostar\b|\bouigo\b|\bsncf\b|\bgare\b"
        r"|\bstation\b|\bperron\b|\bquai\b|\bretard\b|\bvertraging\b|\bdelay(?:ed)?\b"
        r"|\bbillets?\b|\btickets?\b|\bkaartje\b|\bvlucht\b|\bflight\b|\bvliegtuig\b"
        r"|\bavion\b|\bblablacar\b|\bcovoiturage\b",
    ),
    "klimmen & sport": (
        AWAY,
        "|".join(
            # "(?i)" is only allowed at the start of a pattern, so it goes when joined
            [TOPIC_PATTERNS[t].removeprefix("(?i)") for t in ("climbing", "running")]
            + [
                ACTIVITY_TOPICS[t][1]
                for t in ("fietsen", "zwemmen", "yoga", "wandelen")
            ]
        ),
    ),
    "eten & uitgaan": (
        AWAY,
        ACTIVITY_TOPICS["uit eten"][1]
        + "|"
        + ACTIVITY_TOPICS["feest & borrel"][1]
        + r"|\bbar\b|\bd[iî]ner\b|\bdinner\b|\blunch\b|\bbiertje\b|\bbi[eè]re\b|\bbeer\b",
    ),
    "Benji & kind": (HOME, TOPIC_PATTERNS["child"]),
    "wonen": (
        HOME,
        r"\bhuis\b|\bwoning\b|\bappartement\b|\bappart\b|\bflat\b|\bhuur\b|\bloyer\b"
        r"|\bmaison\b|\bhouse\b|\bverhuiz\w*|\bd[ée]m[ée]nag\w*|\bhypotheek\b|\bmortgage\b"
        r"|\bmakelaar\b|\bfunda\b|\bimmobili\w*|\bbezichtiging\w*",
    ),
    "familie": (
        HOME,
        r"\bfamil(?:ie|le|y)\b|\bvader\b|\bmoeder\b|\bouders\b|\bp[eè]re\b|\bm[eè]re\b"
        r"|\bparents\b|\bbroer(?:tje)?\b|\bzus(?:je)?\b|\bfr[eè]re\b|\bs(?:oe|œ)ur\b"
        r"|\boma\b|\bopa\b|\bmamie\b|\bpapi\b|\bschoon(?:ouders|moeder|vader)\b"
        r"|\bbeaux-parents\b|\bbelle-m[eè]re\b|\bbeau-p[eè]re\b|\btante\b|\boom\b|\boncle\b",
    ),
}
HOME_WINDOW = 3  # a "gesprekje": the message with the place and the 3 after it
HOME_MIN_SHARE = 0.08  # lines below this share are left out
# The shuffle test under the picture: Nieuwegein against the three cities of the
# long-distance years, taken together (Roos's sentence, 2026-11-12).
HOME_TEST_PLACE = "Nieuwegein"
HOME_TEST_AGAINST = ("Parijs", "Amsterdam", "Utrecht")
AWAY_COLOR = "#eb6834"
HOME_COLOR = "#1baf7a"
PAPER = "#fcfcfb"

# --- 6.6: Roos and Louis in French, by character trigrams (roos_06.6, 2026-11-12) ----

# Characters per chunk. Louis wrote ~270,000 French characters, so 1,000 gives him about
# 220 chunks; 6.2 used 500 for the whole chat and called it noisy. A starting point.
TRIGRAM_CHUNK_CHARS = 1_000
# The "letters only" check (6.2.5): drop everything that is not a letter, accents kept
# ("ça" stays "ça"), so punctuation, digits, emoji and the keyboard's apostrophe go.
NOT_A_LETTER_PATTERN = r"[^\w ]|[\d_]"
N_LABEL_SHUFFLES = 50  # shuffled author labels for the silhouette baseline, as in 6.2

# --- 6.6.7: a style map from masked trigrams (roos_06.6, 2026-11-12) ---------------
# Moved from scripts/stijlkaart.py on 2026-11-13, when the figure joined the pipeline.

# cleaning:

# WhatsApp's own lines, as they appear in this (Dutch) export. Checked on the chat
# (2026-11-12): calls ("Spraakoproep. Geen antwoord", "Gemiste spraakoproep. Tik om terug te
# bellen"), media ("afbeelding weggelaten"), edits, deletions and the encryption notice.
STYLE_SYSTEM_PATTERNS: tuple[str, ...] = (
    r"\b(?:afbeelding|audio|video|sticker|GIF|document|contactkaart)\s+weggelaten\b",
    r"<Dit bericht is bewerkt>",
    r"(?:Gemiste\s+)?(?:spraak|video)-?oproep\..*",
    r"(?:Je hebt dit bericht verwijderd|Dit bericht is verwijderd)\.?",
    r"Berichten en oproepen worden end-to-end versleuteld\..*",
)
# Phones type ’ or ‘, people type ' : without this, the phone is the biggest "style" difference.
STYLE_APOSTROPHES = "’‘`´"

# French by stopwords:

# Short lists of the most frequent words per language, written out by hand (Roos chose this
# over nltk, 2026-11-12). Words are matched after apostrophes are split off, so "c'est"
# counts as "c" + "est". Only words in exactly one list count (see `unique_stopwords`):
# "de", "je", "en", "la", "me", "que", "tu", "es" and "a" are in two lists and so say nothing.
STYLE_STOPWORDS: dict[str, frozenset[str]] = {
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
STYLE_FRENCH_SHARE = 0.60  # of the recognised stopwords
STYLE_LETTER_RUN = r"[^\W\d_]+"  # a run of letters: "c'est" -> "c", "est"

# units, fragments, masking:

STYLE_UNIT_GAP_SECONDS = (
    5 * 60
)  # a person's French messages closer than this form one unit
STYLE_MIN_UNIT_WORDS = 8
STYLE_FRAGMENT_WORDS = 15
STYLE_MIN_FRAGMENT_WORDS = 8  # a shorter remainder at the end of a unit is dropped
STYLE_EMOJI_PATTERN = "[\U0001f000-\U0001faff\U0001f1e6-\U0001f1ff☀-➿⬀-⯿️‍⃣]"
# A token to mask: letters, possibly joined by hyphens ("est-ce", "peut-être"). Apostrophes,
# punctuation, digits ("0") and spaces stay as typed: the space before "?" is a habit.
STYLE_MASKED_WORD = r"[^\W\d_]+(?:-[^\W\d_]+)*"
STYLE_MASK = "x"
# ~150 French function words: they carry style, not topic. Accentless spellings included,
# because that is how they are often typed.
STYLE_FUNCTION_WORDS: frozenset[str] = frozenset(
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

# quarters, documents, half-years:

# Louis wrote too little French in these quarters to sample from on their own.
STYLE_QUARTER_MERGE = {"2023Q3": "2023Q4", "2025Q2": "2025Q1"}
# The chat ends in Sep 2026: the second half of 2026 is incomplete.
STYLE_HALF_YEAR_MERGE = {"2026H2": "2026H1"}
STYLE_DOCS_PER_QUARTER = 12
STYLE_FRAGMENTS_PER_DOC = 40  # ~600 words
STYLE_MIN_MONTH_FRAGMENTS = 10

# the figure:

STYLE_ROOS_BLUE = "#2a78d6"
STYLE_LOUIS_ORANGE = "#eb6834"
STYLE_PERSON_COLORS = {"Roos": STYLE_ROOS_BLUE, "Louis": STYLE_LOUIS_ORANGE}
STYLE_TEXT = "#1f1f1e"
STYLE_MUTED = "#8a8a83"
STYLE_FRAME = "#e6e6e1"
STYLE_MIN_ARROW_SHARE = 0.025  # of the x range: shorter arrows are skipped

# PCA's sign is arbitrary: an axis whose marker trigrams point the wrong way is flipped.
# x: correct French (low) to loose punctuation and English leftovers (high); y: school
# French (low) to spoken French (high). Taken from the loadings in roos_06.6 §6.6.7.
STYLE_MARKERS: dict[str, dict[str, tuple[str, ...]]] = {
    "x": {"high": (" ;)", " : ", " ! ", " or"), "low": ("doi", "veu", "hez", "peu")},
    "y": {
        "high": (" ca", "ca ", "fau", "oui", " ? "),
        "low": ("bea", "uco", "-ce", " n'"),
    },
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

    @property
    def weekly_french(self) -> Path:
        return self.processed / "3_weekly_french.parquet"

    @property
    def french_stats(self) -> Path:
        return self.processed / "3_french_stats.json"

    # Figures 7 and 8 start from the messages split back apart (UnmergeMessages), a
    # second stage-1 file, so figures 1-6 keep the rows they were made with.
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

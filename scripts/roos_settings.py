"""The notebooks' settings: flat names, as `roos_pipeline/settings.py` had them before 2026-11-13.

The pipeline's settings became pydantic classes per kind (`TIMELINE`, `PATTERNS`, ...),
and `roos_pipeline/` keeps only what the six final figures need. The notebooks of lessons
4-7 still read `settings.MOVE_DATE`, `settings.PLACES`, ...: they import this module
instead (`from scripts import roos_settings as settings`), so their cells stay as they
were.

One answer per value: everything the pipeline also uses (dates, regexes, colours, the
shuffle's settings) is taken from the pipeline's classes below, not copied. Only what no
pipeline figure uses is written out here.
"""

from __future__ import annotations

import pandas as pd

from roos_pipeline.settings import (
    CATEGORIES,
    COLORS,
    GROUPING,
    LIMITS,
    PATTERNS,
    TESTS,
    TIMELINE,
    Labelled,
    Paths,
    Side,
)

__all__ = ["Labelled", "Paths", "Side"]  # re-exported for the notebooks

# --- shared with the pipeline: one answer, taken from roos_pipeline/settings.py ------

MOVE_DATE = TIMELINE.move
TEST_DATE = TIMELINE.test
BIRTH_DATE = TIMELINE.birth
PREGNANCY_START = TIMELINE.pregnancy_start
PERIOD_ORDER = list(CATEGORIES.periods)

MIN_CHARS = LIMITS.min_chars
TOPIC_PATTERNS = PATTERNS.topics
SYMPTOM_PATTERNS = PATTERNS.symptoms
LINK_PATTERN = PATTERNS.link

HOME_PLACES = PATTERNS.home_places
HOME_THEMES = PATTERNS.home_themes
HOME_WINDOW = GROUPING.home_window
HOME_MIN_SHARE = LIMITS.home_min_share
PAPER = COLORS.home.paper

SMOOTH_WEEKS = GROUPING.smooth_weeks
N_SHUFFLE = TESTS.symptoms.n_shuffle
SEED = TESTS.symptoms.seed

GREY = COLORS.bars.grey
LIGHT_GREY = COLORS.bars.light_grey
DARK_GREY = COLORS.bars.dark_grey
LONG_DISTANCE_ORANGE = COLORS.familiar.dots
TOGETHER_BLUE = "#0072B2"  # roos_05.6: Okabe-Ito blue next to the orange (Roos's pick)

# --- lesson 5: Louis's messages (roos_05.6) ---------------------------------------

# A reply that quotes an earlier message carries its "[18-12-2025, 09:42:27] Roos: ..."
# line, so the quoted words count as the sender's (found in roos_05.6, 2026-10-09).
QUOTED_REPLY_PATTERN = r"\[\d{1,2}-\d{1,2}-\d{4}, "

# --- lesson 6: Louis in French vs. English (roos_06.5) ----------------------------

# A "word" for the group keywords (6.5): 3+ letters, no digits, so "je", "ok" and "18:51" never count.
KEYWORD_PATTERN = r"(?u)\b[^\W\d_]{3,}\b"
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

# --- 6.5.10: topics from word lists (roos_06.5) -----------------------------------

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

# --- 6.6: Roos and Louis in French, by character trigrams (roos_06.6) -------------

# Characters per chunk. Louis wrote ~270,000 French characters, so 1,000 gives him about
# 220 chunks; 6.2 used 500 for the whole chat and called it noisy. A starting point.
TRIGRAM_CHUNK_CHARS = 1_000
# The "letters only" check (6.2.5): drop everything that is not a letter, accents kept
# ("ça" stays "ça"), so punctuation, digits, emoji and the keyboard's apostrophe go.
NOT_A_LETTER_PATTERN = r"[^\w ]|[\d_]"
N_LABEL_SHUFFLES = 50  # shuffled author labels for the silhouette baseline, as in 6.2

# --- lesson 7: places, people and topics (roos_07.4) ------------------------------

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
    "Utrecht": ("Netherlands", PATTERNS.home_places["Utrecht"]),
    "Amsterdam": ("Netherlands", PATTERNS.home_places["Amsterdam"]),
    "Rotterdam": ("Netherlands", r"\brotterdam\b"),
    "Zoetermeer": ("Netherlands", PATTERNS.home_places["Zoetermeer"]),
    "Nieuwegein": ("Netherlands", PATTERNS.home_places["Nieuwegein"]),
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
    "fietsen": ("sport", PATTERNS.activities["fietsen"]),
    "zwemmen": ("sport", PATTERNS.activities["zwemmen"]),
    "yoga": ("sport", PATTERNS.activities["yoga"]),
    "wandelen": ("sport", PATTERNS.activities["wandelen"]),
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
    "uit eten": ("uit & thuis", PATTERNS.activities["uit eten"]),
    "feest & borrel": ("uit & thuis", PATTERNS.activities["feest & borrel"]),
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

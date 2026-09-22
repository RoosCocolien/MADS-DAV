# Analysis log — own chat (Roos & Louis)

Plain-text home for the `goad` coaching plan, since the MCP server itself holds no
state between VS Code restarts. Append to this after every stage; refer back to it
any time you reopen the project.

## Context (stage 1 — question)

- **Data**: full relationship history in one chat, from meeting (long-distance
  Paris–Amsterdam) through moving in together in Utrecht, a pregnancy, a birth, and
  a temporary separation for a course. Two authors: Roos and Louis.
- **Known timeline** (from memory, not yet verified against the data):
  - Aug 2023: met (long-distance Paris–Amsterdam)
  - Summer 2024: pregnancy
  - Nov 2024: Louis moves to Utrecht (small apartment together) — *see open question below,
    one source said Oct 2024*
  - May 2025: Benji born
  - Mar 23 2025 – Oct 6 2025: Roos's maternity leave
  - Jun 16 2025: Louis starts at MetNils (current job)
  - Aug 2026: Louis's intense 4-week Dutch course in Belgium (temporary separation again)
- **Suspicions going in**: Roos's own messages became longer, more frequent, and more
  planning/logistics-focused since Benji was born (she does the finances). Love
  messaging felt like it stayed intact, maybe fewer in absolute count since Louis
  moved to NL but still a large relative share.
- **Boring/expected result**: message volume drops after Louis moves to NL; love-message
  share stays roughly constant; planning/logistics messages increase.
- **What would actually surprise her**: if the relative share of love messages dropped.
- **Response time suspicion**: not a fixed person difference — expects Louis's response
  time to track his job situation, and her own to be slower during maternity leave.
  The opposite of either would be the surprising result.
- **Categories are hers**, not data-mined — topics she already considers important in
  the relationship, not patterns noticed by skimming.
- **Audience**: Lesson 1, exercise 2 ("Draft visualisation 1: comparing categories,
  before Lesson 2") — a draft, not the graded Lesson 2 submission.

## Stage 2 — data

- **One row** = one message (timestamp, author, message text). Confirmed.
- **Data source resolved**: `config.toml`'s `current` now points at
  `whatsapp-20260911-122106.csv`, whose first message is **2023-08-12** — i.e. this
  export covers the full history back to when they met. The earlier `01.4` notebook's
  note about a Nov-2024 start was about an *older* export; that concern no longer
  applies with this file.
- **n=2 for any per-person claim.** Every "who responds faster / sends more" comparison
  is between exactly two people — descriptive, not something a p-value helps with.
- **Response times are not independent draws.** A fast back-and-forth produces several
  correlated short response times from the same conversational moment, not several
  independent measurements of "typical" responsiveness. Plan: report the **median**
  (resists both the long overnight tail and a chatty-evening cluster) alongside the
  full histogram, rather than collapsing to a mean.

### Time-of-day buckets

- Morning: 05:00–12:00
- Afternoon: 12:00–17:30
- Evening: 17:30 and later

### Louis's job periods (for response-time-by-period)

| Period | Situation |
|---|---|
| until Apr 2024 | architectural agency (when they met) |
| Apr 2024 – Oct 2024 | start-up in Paris, disliked |
| Nov 2024 | moves to the Netherlands (confirmed: start-up ended Oct 2024, move was Nov 2024) |
| after move – early 2025 | job-hunting |
| early 2025, ~3 weeks | start-up in Amsterdam, disliked |
| gap | unspecified |
| 16 Jun 2025 onward | MetNils (current) |
| Aug 2026 | intense Dutch course in Belgium (temporary) |

### Roos's maternity leave

- 23 Mar 2025 – 6 Oct 2025

### Category features (7 + 1), final — implemented in `01.4-language-exploration.ipynb`

Eight `RegexFeature` steps, French/Dutch/English, case-insensitive:

1. **Love** — je t'aime, tu me manques, mon amour, mon cœur, ma belle, mon chéri(e),
   je pense à toi, bisou(s), t'es belle/beau; ik hou(d) van je/jou, ik mis je/jou,
   schatje, liefje; I love you, I miss you, my love, sweetheart, darling; heart/kiss
   emoji. **"bébé" deliberately excluded** — moved entirely to Child (see below).
2. **Financial** — hypotheek(adviseur)/hypothèque/mortgage(advisor); rekening/compte/bank
   account; belasting/impôts/tax(es); salaris/salaire/salary/wage; huur/loyer/rent;
   borg/caution/deposit; spaarrekening/savings account; geld/argent/money;
   betalen/payer/pay; overmaken/virer/transfer; bunq, asn, ing, rabobank, funda,
   velduil, notaris/notaire/notary, makelaar/agent immobilier/real estate agent, iban,
   factuur/facture/invoice, budget.
3. **Child** — benji, bébé/baby, crèche/kinderdagverblijf/opvang/daycare, enfant,
   biberon/bottle-feeding, sucette/fopspeen/speentje/pacifier/dummy, allaiter/donner le
   sein/borstvoeding/breastfeeding/tétée, luier(s)/diaper(s), doekje/hydrophile/
   hydrofiele doek/wipe, kinderwagen/poussette/stroller/pram, la remorque/bike trailer,
   consultatiebureau, kraamzorg, zwangerschap/grossesse/pregnancy/pregnant,
   verloskundige/sage-femme/midwife, kolven, échographie.
4. **Climbing** — escalader/escalade, salle de grimpe, grimper/grimpe, climb(ing),
   climbing gym, klimmen, klimmuur (centraal), boulder, bloc, voie, arkose, mountain
   network, ncc, neoliet, grades 5c–8a+.
5. **Running** — run(ning), courir, trail, borne(s), kilomètre, km, marathon, hardlopen,
   hardloopschoenen, joggen, parkrun, 10k, tempo, chaussures, shoes.
6. **Food** — eten/manger, dîner/avondeten/dinner, déjeuner/lunch, petit-déjeuner/
   ontbijt/breakfast, cook(ing)/cuisiner/koken, recette/recept/recipe, boodschappen/
   courses/groceries, restaurant, honger/faim/hungry.
7. **Planning & logistics** — quand/when, je suis là, j'arrive/arriver, tu rentres
   quand, je rentre, en route, à quelle heure/hoe laat, "I'll be there", thuis/home,
   onderweg/on the way, trein/train, vlucht/flight, agenda, rendez-vous/afspraak/
   appointment.
8. **Be careful / safety** — fais attention, fais gaffe, sois prudent(e), attention à
   toi, prends soin de toi, take care, be careful, pas op, wees voorzichtig, let op.

**Resolved overlap**: "bébé" stays only in Child; Love relies on the endearment phrases
instead. Flagged as a known false-positive risk: climbing grades (`5c`–`8a+`) are short
alphanumeric tokens and the category most worth spot-checking.

## Features implemented (stage 2 technical output)

- `response_seconds` — gap to the next message from the *other* author; same-author
  bursts get `NaN` (not counted as a "response").
- `time_of_day` — morning 05:00–12:00, afternoon 12:00–17:30, evening 17:30+.
- `n_chars`, `n_words` — message length.
- 8 category counts, `n_<category>_words`.

## Resolved open questions

- Louis moved to Utrecht **Nov 2024** (start-up in Paris ended Oct 2024, move was Nov).
- "bébé" → Child only, dropped from Love.
- Category word lists reviewed and finalized (see above).

## Stage 3 — shape

- Response time: expected lopsided, median over mean confirmed.
- Message length: also lopsided, plus a small number of notably long messages — median
  for "typical," a separate look at the long-message outliers.
- Category counts: raw per-message shape doesn't matter; the weekly *aggregated*
  series is what matters. Weekly, not monthly (too coarse a stretch).
- n=2 reaffirmed: every result describes this one relationship, not people in general.

## Stage 4 — encoding

- Family: time (a line per category), not a ranking/comparison — the question is
  about development over the relationship's timeline.
- First sketch: all 8 categories, one shared-axis plot, raw weekly counts,
  `palette="colorblind"`. Not split by person yet.
- Long-term goal named: show planning/logistics becoming a bigger part of the
  relationship as it matured — flagged in advance that this needs *shares* (÷ weekly
  total messages), since raw volume itself isn't stable (dropped after the move).

## Stage 5 — critique (of the first shared-axis plot)

Read by the student, category by category:

- **Food**: expected stable; wants a relative (share) view — also finds it genuinely
  interesting either way (does food talk shrink, or just ride the same volume drop as
  everything else after the move?).
- **Child**: flat pre-pregnancy, an intense peak during it, a drop around Benji's birth
  (May 2025) alongside everything else dropping too — matches her expectation.
- **Climbing**: a hot topic early in the relationship, fading during the pregnancy —
  matches her expectation.
- **Running**: no visible increase in the shared-axis plot — inconclusive, not
  necessarily false; flagged as possibly hidden by the shared y-axis.
- **Planning & logistics**: the actual surprise — looked fairly even, not rising,
  which *contradicts* her own stage-1 "boring/expected" prediction that it would
  increase.
- **Be careful**: too thin (127 hits total) to hold its own against the rest — cut
  from this comparison chart.
- **Financial**: lost visually among the other 7 colours on a shared axis.

**Fix applied**: small multiples (own y-scale per category, `be_careful` dropped) +
a share-of-weekly-total-messages version, both built in the notebook.

**New observation from the fixed charts, not yet interpreted with the student**:
the share view suggests `planning_logistics` may actually be *declining* rather than
flat, and early weeks show large spikes/instability in every share plot (`food`
share even exceeds 1.0 at points) — likely a small-denominator artifact from weeks
with very few total messages just after they met, not a real surge. Open for her to
read next, same as before — not asserted as a conclusion here.

## Still open

- [ ] Interpret the small-multiples raw + share charts (student's read, not assistant's).
- [x] Milestone lines (pregnancy/move/baby born) + readable date ticks added to both
      small-multiples charts.
- [x] Share metric corrected: it was summed word-count ÷ weekly messages (could exceed
      1 and was misleadingly labelled "share"); now it's the fraction of that week's
      messages that mention the category at least once, properly bounded [0, 1].
- [x] Language mix before vs. after the move (Nov 2024), as two side-by-side percentage
      bar charts, `unknown`-language messages excluded.
- [x] Packed-bubble overview of all 8 categories (area, not radius, = share of all
      messages mentioning the topic) — deliberately artsy, not a rigorous claim.
- [ ] Person-split version, once the unsplit overview has said enough.
- [ ] Response time, message length, and volume-per-person charts — not yet built.
- [ ] Stage 6 (verification) once a chart is trusted enough to defend.
- [x] Bubble chart redesigned: spiral circle-packing (guarantees every bubble fits in
      frame, unlike the earlier rejection-sampling version), Okabe-Ito colorblind-safe
      palette with `love` pinned to pink/purple, emoji via `family="Segoe UI Emoji"`
      (renders as outline glyphs — matplotlib can't do full-colour emoji), off-white
      background for an "artsy" feel. Revised again: icon now centered in each bubble
      (no more offset label text), a proper legend replaces on-bubble names, be_careful's
      white icon sized down to stay visibly smaller than its (already tiny) black bubble,
      title "Liefde gaat door de maag" added.
- [x] New chart: 5 merged categories (`sport` = climbing∪running, `financial_planning`
      = financial∪planning_logistics, `be_careful` dropped) × 3 relationship periods
      (before move / after move–before baby / after baby), grouped bar chart of
      per-period share. **Open question sent back to the student**: "in pairs" was
      read as a grouped bar chart (5 adjacent bars per period), not a Roos-vs-Louis
      split — needs her confirmation.
- [x] Confirmed: student wants both — added a second version of that same chart, split
      Roos vs. Louis (share of each person's *own* messages that period, not the
      couple's combined total, so volume differences don't distort the comparison).
      `author` is anonymised in the data (`humanize()` from 01.3) — relabelled to real
      names for this personal chart via the existing `canonical_author`/`anon` mapping.

## Session — 2026-09-17

- **Standing instruction**: update this log after every session working together, not
  only at the end of a `goad` stage.
- **Cell 12 of `01.4-language-exploration.ipynb`** ("When parenthood enters the chat")
  redesigned from an 8-panel small-multiples grid to a single combined chart, scoped
  down to just `child` and `climbing` — the two categories that actually move opposite
  each other. Choices made, for the student to defend:
  - Both categories share one axes instead of separate panels, so the trade-off between
    them is visible directly rather than inferred by eye across two small subplots.
  - The raw weekly share is kept as a thin light-grey line (per the "keep the real
    line" instruction) with a 4-week centered rolling-mean trend line drawn over it in
    each category's colour (`child` blue `#0072B2`, `climbing` orange `#D55E00`) —
    **assumption to confirm**: 4 weeks was picked as a reasonable smoothing window,
    not requested as a specific number; revisit if the trend looks over- or
    under-smoothed.
  - The pregnancy → baby-born span is shaded (light yellow, low alpha) behind the
    lines, in addition to the existing dashed milestone lines.
  - A caption below the plot states the n behind each line: **assumption to
    confirm** — this reports total messages that ever mention each category
    (`child: n=2,218`, `climbing: n=1,578`) over the whole period, not a per-week or
    per-conversation count. Directly answers the earlier spurious-correlations
    feedback ("vermeldt hoeveel gesprekken dit op gebaseerd is") — check with the
    student that message-count is the right unit, versus conversation-count.
  - Verified by truncating the notebook to cells 0–21 and executing with
    `jupyter nbconvert --execute`: runs clean, produces the chart, no errors.
- **Notebook workflow fix**: `references/tooling-setup.md` (M5) says never edit a
  teacher notebook in place — copy it with a name suffix first. Four notebooks had
  been edited in place (`00.1-classes.ipynb`, `01.1-goad-toolkit-101.ipynb`,
  `01.2-irc-chat.ipynb`, `01.3-your-own-chat.ipynb`). Made `roos_`-prefixed copies of
  all lesson 1 and lesson 2 notebooks (student's naming choice; course docs suggest a
  suffix instead) with the current edits preserved, so the originals can be reverted
  and `main` pulled cleanly. Lesson 0's `00.1-classes.ipynb` was flagged but not yet
  copied — pending student's call.
- **Root-level clutter resolved**: dozens of `_Users_...ipynb.json` /
  `.ipynb.log` files in the repo root turned out to be `notebooktester` result-cache
  entries (same schema as the real `.notebookcache/` files), written to the repo root
  instead — traced to the editor's background notebook-validation, not to
  `.lefthook.yml`'s hook (which correctly runs `notebooktester notebooks -v -t 150`
  from the repo root). Deleted the stray files and added `*.ipynb.json` to
  `.gitignore` next to the existing `.notebookcache` entry so future ones don't end up
  staged by accident.
- **Found while testing**: `notebooktester` currently fails on
  `01.4-language-exploration.ipynb` with a Windows `'charmap' codec` decode error,
  unrelated to today's edit — traced to a later cell (the emoji-based packed-bubble
  chart, `family="Segoe UI Emoji"`) hitting a non-UTF-8 console encoding crash on
  Windows. A stray raw Python traceback is also sitting in cell 24 as if it were
  markdown content — leftover debugging output, not a real markdown cell. Not fixed
  yet; flagged for the student to decide how to handle (drop the emoji, or set
  `PYTHONUTF8=1`).
- **Follow-up**: the student committed the `roos_`-prefixed copies herself in two
  commits (`c9643b6` lesson 1, `27f51e2` lesson 0 — the latter includes
  `roos_00.2-paths copy.ipynb`, a stray " copy" left in the name from a paste
  operation rather than the `roos_` convention; worth a rename later) and reverted
  the in-place-edited originals ahead of the assistant doing so.
  `01.4-language-exploration.ipynb` (never a teacher file, always untracked) no
  longer exists — `roos_01.4-language-exploration.ipynb` is now the only copy,
  committed *before* the cell 12 redesign above landed. Ported the same cell 12
  rewrite into `roos_01.4-language-exploration.ipynb` (same cell id `1b7f6c09`),
  verified by truncating to cells 0–21 and executing — byte-identical output to the
  first version. Also cleaned up: the same stray `notebooktester` result files
  inside `notebooks/lesson1/`, and a `notebooks/lesson1/logs/` directory that was
  the assistant's own byproduct from running `notebooktester -v` there earlier —
  neither is anyone's work, both deleted.
- **Confirmed**: the cell 12 n-caption should count messages (not conversations) —
  matches what was already implemented, no code change needed.

## Session — 2026-09-18: rebuilding "Praat Nederlands met mij" for lesson 2

Goad six-stage interview, started from `roos_02.5-your_turn.ipynb` ("your turn"
exercise), targeting the teacher's feedback on `roos_01.4`'s categories submission
(published 2026-09-15, provisional grade 8/10).

### Stage 1 — question

- **Target chart**: `categories_13_praat_nl_met_mij.png` (`roos_01.4`, cell 23) — two
  side-by-side bar panels, language mix (en/nl/fr) before vs. after the move to NL, for
  the couple combined. This is the chart the teacher's `preattentive-gestalt` comment
  ("2x3 dingen vergelijken, zet de barren naast elkaar") was about — literally 2 periods
  × 3 languages.
- **Proposition**: after the move, Louis's share of Dutch messages grows while Roos's
  stays dominantly French — an asymmetric shift, not both partners moving together. Any
  other outcome (symmetric shift, or no shift for Louis) would surprise her.
- **Boring/null result named**: both people increasing Dutch roughly in step after the
  move — "everyone adapts to the local language," nothing person-specific to say.
- **Arc/motivation**: not neutral curiosity — the chart doubles as a personal nudge: if
  the move alone hasn't already pulled Louis toward Dutch, that's the argument for Roos
  to speak more Dutch with him herself so he learns faster.
- **Origin**: prior hunch about the relationship (she already suspects herself of
  "stubbornly" staying in French), not spotted while skimming the aggregate chart.
- Answers the `verhaallijn` feedback ("wie zegt welke taal tegen wie") directly.

### Stage 2 — data

- Needed columns already exist or were added: `language` (`py3langid`, restricted to
  en/nl/fr, `LanguageFeature` step, `min_chars=3` threshold below which a message is
  `"unknown"`), `period` (before/after `MOVE_DATE` 2024-11-01), `author`.
- **Fix applied in `roos_02.5-your_turn.ipynb`** (cell `c03`, the loading cell): Louis
  appears under three export names (`Roos`, `~Louis Chessé`, `Louis`, `Louis Ancien`) —
  added `canonical_author()` right after `load_own_chat()` so every downstream cell in
  that notebook sees one `"Louis"`.
- **`unknown` ≠ classifier uncertainty** — `LanguageFeature` marks a message `unknown`
  purely on `len(message.strip()) < 3`, regardless of content.
- **Unknown rate by author × period**: Louis 1.01% (before) → 3.01% (after move),
  roughly triples; Roos 1.05% → 1.15%, flat.
- **Sampled Louis's 296 after-move `unknown` messages**: 134 are `"Ok"`/`"ok"` (English
  filler, not Dutch), 20 `"Ja"`/`"ja"` + 9 `"Ik"` (29 genuine Dutch tokens lost to the
  cutoff), ~65 emoji-only, rest misc (numbers, punctuation).
- **Student's read**: sub-3-character messages don't really reflect language practice
  either way — keep dropping `unknown` from the percentages (no change), but add
  explicit text stating the 3-character cutoff so a reader knows what's excluded and
  why.
- **Message-volume asymmetry raised**: Roos sends more messages than Louis overall
  (before: Louis 18,149 / Roos 20,353; after: Louis 9,844 / Roos 15,090 — Louis's
  volume nearly halves after the move). Resolved: both planned charts use each
  person's own share (% of that person's own messages that period), so the volume
  difference doesn't bias the comparison itself — but the per-period n (esp. Louis's
  smaller post-move n) still belongs in the same n-caption the `spurious-correlations`
  feedback asked for.
- No new `Pipeline` steps needed — the person × period aggregation stays notebook-cell
  code (exploratory/display logic, not reusable feature engineering).

### Plan for the two charts (encoding, agreed so far)

1. **Plot 1** — combined couple view, redone as one axes of grouped bars (x=language,
   hue=period) instead of two side-by-side panels — the direct gestalt fix. Built with
   `BarPlot` from `scripts/plots.py` (`hue=` passes through `**kwargs` to
   `sns.barplot`, no new class needed).
2. **Plot 2** — person-split, as a *difference* per language (Louis's % − Roos's %),
   split by period (before/after → 6 bars: 3 languages × 2 periods) rather than
   parallel bars per person.

### Still open

- [ ] Build Plot 1 and Plot 2 in `roos_02.5-your_turn.ipynb`.
- [ ] Add the 3-character-cutoff methodology text.
- [ ] Add the n-caption (messages per person × period).
- [ ] Stage 3 (shape) and stage 4 (encoding) formal sign-off, stage 5 (critique) once
      charts exist, stage 6 (verification).

## Session 2026-09-21 — `roos_03.3-events-in-your-chat.ipynb`

- **Bug fixed (cell 3):** `own["timestamp"]` arrives as text from the CSV, so
  `.resample("D")` raised `TypeError: Only valid with DatetimeIndex`. Fix:
  `own["timestamp"] = pd.to_datetime(own["timestamp"])` right after `load_own_chat()`.
- **New section 3.3.5 ("the *how* row")**, ported from `roos_01.4`: the `child` and
  `climbing` categories (`CATEGORY_PATTERNS`, unchanged; one `RegexFeature` per category
  in a `Pipeline`). Counts reproduce 01.4: child 2,218 / climbing 1,578 messages.
- **Weekly share plot** — `LinePlot` (grey raw weeks, coloured 4-week `RollingAvg`) plus a
  `VerticalDate` per entry in `my_events`. Note `RollingAvg` is trailing, not centred like
  the 01.4 version.
- **Per-period bars** — `EventPeriod` (a `TransformBase` subclass defined in the notebook)
  labels each message by the last event before it; drawn with `BarPlot` from
  `scripts/plots.py`.
- **Prediction left blank on purpose** — the markdown cell asks Roos to write it before
  running the plots (the point of the notebook).

### Still open

- [ ] Write the prediction in 3.3.5 *before* reading the two new plots.
- [ ] Decide whether the per-period split (four periods from `my_events`) is the
      comparison that matches that prediction.
- [ ] Section 3.3.7: what was falsifiable, and one impact that could not be found.

## Session 2026-09-22

- **`codestyle` MCP server was failing to connect**: `ModuleNotFoundError: No module
  named 'mcp.server.mcpserver'`. Traced to a version conflict, not a missing package —
  `codestyle_mcp.py`'s own PEP 723 header now pins `mcp[cli]>=2,<3`, but `.mcp.json`'s
  launch command forced `uv run --with "mcp<2"` on top of it, so uv resolved an `mcp<2`
  install that predates the `mcp.server.mcpserver` module the script imports.
  **Fix applied**: dropped the `--with "mcp<2"` override in `.mcp.json` so `uv run` uses
  the script's own declared dependency instead. Verified by launching the server
  directly (`CODESTYLE_REF=stable uv run --no-project https://raw.githubusercontent.com/raoulg/codestyle/stable/codestyle_mcp.py`) — starts clean, no traceback. This is a
  repo-wide file merged from the teacher's `origin`, so a future upstream merge could
  reintroduce the stale pin; worth flagging to the class/teacher rather than treating as
  fully resolved.
- **New section 3.3.8 in `roos_03.3-events-in-your-chat.ipynb`**: a `nausea` category
  (English/Dutch/French `RegexFeature`, same shape as `child`/`climbing`) plotted as a
  weekly line (grey raw + 4-week `RollingAvg`, reusing `SMOOTH_WEEKS`) with three
  `EventMarker`s from `scripts/plots.py` — positive pregnancy test, a **computed**
  pregnancy-start date (`Benji's birth − 41 weeks = 2024-07-28`, the standard
  last-period due-date convention, not a conception date), and Benji's birth.
  **n=38 messages mention nausea** — thin next to `child` (2,274) or `climbing`
  (1,764), flagged in the reflection cell as a signal to sanity-check per-week before
  trusting any rise in the line.
  **Prediction and reading left blank on purpose**, same convention as 3.3.5 — the
  markdown cells ask Roos for her prediction and her read of the chart before either is
  filled in; not answered here.
  Verified by executing the full notebook with `jupyter nbconvert --execute --inplace`
  — runs clean, no errors.

- **Prediction recorded (3.3.8)**, in the student's own words: nausea messages expected
  to start around week 5 of the pregnancy and disappear around week 16. Turned into
  calendar dates off the same `pregnancy_start` anchor (2024-07-28): week 5 =
  2024-09-01 (coincides exactly with the existing "positive pregnancy test" marker, so
  no separate marker drawn for it), week 16 = 2024-11-17 — added as a fourth,
  purple-coloured `EventMarker` ("predicted nausea ends") so the guess sits on the same
  chart as the confirmed timeline, distinguishable by colour. Re-executed clean via
  `jupyter nbconvert --execute --inplace`.
  **Not yet answered**: whether the actual line matches the prediction — left for the
  student to read against the n=38 caveat already in the notebook.

- **New section 3.3.9**: re-scoped the nausea chart to a narrower question — pregnancy
  window only (`pregnancy_start` to Benji's birth), Roos's own messages only, raw
  mention counts (`n_nausea_words`, not a share) on the y-axis, x-axis switched from
  calendar dates to week-of-pregnancy (week 1 = first week after `pregnancy_start`,
  through week 41, missing weeks filled to 0 rather than dropped).
  **Bug hit and fixed**: `own_nausea["timestamp"]` is tz-aware (`UTC`) but
  `pregnancy_start`/`pregnancy_end` are naive `Timestamp`s — direct comparison raised
  `TypeError: Invalid comparison between dtype=datetime64[us, UTC] and Timestamp`. Fixed
  the same way the notebook's own `EventPeriod` class already does it: drop the tz on a
  copy of the timestamp column (`.dt.tz_localize(None)`) rather than localising the
  milestone dates to a timezone they were never given.
  **Result**: 9,874 of Roos's messages fall inside the pregnancy window; 30 of the
  dataset's 38 total nausea mentions are hers. Re-executed clean via
  `jupyter nbconvert --execute --inplace`.

- **New `RiseDeclinePlot` class added to `scripts/plots.py`**, at the student's request
  ("use the base classes from script, add a subclass if necessary") — neither `BarPlot`
  nor goad_toolkit's `LinePlot` encode *direction*. Draws a line whose segments are
  coloured by whether the value rose or held/fell since the previous point
  (`settings.highlight_color` / `settings.base_color`, grey-first per 02.2), with a
  two-entry legend (not one per segment) built from `Line2D` proxies. Flat runs keep
  the prior segment's direction rather than flipping colour with no real change.
- **New section 3.3.10**: a `fatigue` category (English/Dutch/French `RegexFeature`,
  word list flagged in the notebook for the student's own review — `\bmoe\b` is
  boundary-matched against Dutch "moeder"/"moeite" but therefore also misses compounds
  like "doodmoe"), same scope as 3.3.9 (Roos-only, pregnancy window, weekly counts,
  week-of-pregnancy x-axis), plotted with the new `RiseDeclinePlot`.
  **Result**: 584 messages mention fatigue dataset-wide; 157 of Roos's mentions fall
  inside the pregnancy window — a much bigger n than nausea's 30, but also a much more
  generic word set ("tired"/"moe" vs. "nausea"), so more of these hits are plausibly
  ordinary tiredness rather than a pregnancy-specific signal — flagged, not resolved.
  Re-executed clean via `jupyter nbconvert --execute --inplace`.

### Still open

- [ ] Read all three symptom charts (3.3.8 nausea/calendar, 3.3.9 nausea/week,
      3.3.10 fatigue/week) against predictions and fill in each "What I found."
- [ ] Review the `fatigue` word list — is generic tiredness noise acceptable, or does
      it need narrowing (e.g. requiring co-occurrence with other pregnancy language)?
- [ ] Decide whether to report the `.mcp.json` `codestyle` pin bug upstream/to the
      teacher, since other students on `origin` likely hit the same error.

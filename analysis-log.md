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

## Session 2026-09-22 (cont'd) — new notebook, `roos_04.6-distribution.ipynb`

- **Created `notebooks/lesson4/roos_04.6-distribution.ipynb`** — a new "your turn"
  notebook, not a copy of a teacher file (lesson4 has no 04.6). Opening structure
  mirrors `roos_03.3` and `roos_02.5`: intro, standard `goad_toolkit` distribution
  imports (`DistributionRegistry`, `DistributionFitter`, `fit_table`, `HistogramPlot`,
  `DistPlot`, `QQPlot`, `ECDFPlot`, `PlotFits`) plus the `scripts/plots.py` base classes,
  `load_own_chat()` + `canonical_author()` setup, a "what's already loaded" section, and
  a `goad` brainstorming section naming `goad_distribution_method` specifically.
  **No analysis content yet, deliberately** — the student asked to be guided by
  questions rather than handed a finished notebook, so the last cell is a blank "Your
  turn" prompt (variable, independent-observation count, discrete/continuous,
  predicted family — `goad_distribution_method`'s own step order) for her to answer
  before any fitting happens, matching CLAUDE.md's coaching policy directly. Verified
  by executing the setup cells (`jupyter nbconvert --execute --inplace`) — imports and
  `load_own_chat()` run clean.
- **Candidate variables discussed in chat** (not chosen yet): response time
  (`response_seconds`, already computed in `roos_01.4`), message length (`n_chars`),
  daily message counts, inter-message gaps, and word/emoji rank-frequency (a Zipf/
  power-law candidate). Presented as options, not a recommendation — the student picks.

- **Student's answers recorded** in the "Your turn" cell: response time, split
  Roos/Louis × before/after Louis's move to NL (Nov 2024); predicted outcome (who's
  faster, before vs. after) but no family committed yet; flagged the overnight-sleep
  problem herself before being asked ("at night we sleep").
- **Bug fixed in `roos_04.4-before-and-after.ipynb`, cell `In[12]` (id `e2ce92a5`)**:
  `StopIteration` from `next(f for f in fits if ...)`. Root cause was upstream, in cell
  `b63ba843`: `my_event = 2024-11-11` — missing quotes, so Python evaluated it as
  `2024 - 11 - 11 = 2002` (an int) instead of a date string. `pd.Timestamp(2002, ...)`
  then landed near the Unix epoch, so every message counted as "after" it, the "before"
  bucket had 0 days, every discrete-family fit on an empty array failed, and `next()`
  over an all-failed generator raised `StopIteration`. Fix: added the missing quotes
  (`my_event = "2024-11-11"`). Verified by re-executing the full notebook — clean.
- **Section 4.6.4 built in `roos_04.6-distribution.ipynb`**: `response_seconds`
  (`ResponseTimeFeature`, reused unchanged from `roos_01.4`) × `MOVE_DATE` (reused from
  `roos_02.5`, 2024-11-01) → four groups (Roos/Louis × before/after). **New assumption,
  not yet confirmed by the student**: responses over `MAX_RESPONSE_MINUTES = 240` (4h)
  are excluded as "asleep, not a slow reply" — more generous than 04.4's own 1-hour
  cutoff for a bare message gap, since a real response can take longer than an hour
  without being an overnight gap; 786 responses excluded on this run. Plotted as four
  `ECDFPlot` curves on one axes (colour = person, linestyle = before/after) — no
  distribution fitted yet, per the student's "first I want the first one" (ECDF over
  fit+null). Sample sizes: Roos before n=7,361, after n=4,546; Louis before n=7,470,
  after n=4,562. Re-executed clean via `jupyter nbconvert --execute --inplace`.

### Still open

- [ ] Student reads the four-ECDF chart in `roos_04.6-distribution.ipynb` against her
      stated prediction (lower response time before the move for her, faster for Louis,
      possible switch after).
- [ ] Confirm or adjust `MAX_RESPONSE_MINUTES` (currently 4h, assistant's default).
- [ ] Decide whether to move on to fitting a family + `NullDistribution` (the "second
      one" from the assistant's original two-option question), once the ECDF has been
      read.

## Session 2026-09-22 (cont'd) — 4.6.5, hour-of-day before/after the move

- **New hypothesis from the student**: before the move to NL, she and Louis exchanged a
  long-distance "good morning" / "sleep well" ritual by text; she expects it to
  disappear once they share an apartment. Explicitly framed as a known event with a
  named prediction, not event- or pattern-discovery — confirmed with her that 03.3
  (event-hunting) and 03.2 (autocorrelation/pattern-discovery) don't apply here.
- **Three options presented** (greeting-word `RegexFeature`, hour-of-day distribution,
  or both combined); **student picked hour-of-day** as the first build. Greeting-word
  frequency and the combined (greeting-word-hours-only) version are still open, and are
  the more direct test of the actual "we stop saying the words" claim — hour-of-day
  alone only shows whether the day's *rhythm* changed, which could have other causes.
- **Section 4.6.5 built**: hour extracted via `TimeFeatures`, night hours (00:00-04:59,
  matching this log's existing "morning starts at 05:00" boundary) excluded, each
  period's hours expressed as a **share of that period's own messages** (not raw
  counts — same unequal-group-size caveat as the language-mix and response-time work).
  Grouped `BarPlot` (x=hour, hue=period), grey-first: "before move" in
  `settings.base_color`, "after move" (the change) in `settings.highlight_color`.
  Re-executed clean via `jupyter nbconvert --execute --inplace`.

- **Prediction recorded in the notebook** (written into 4.6.5, before the "what I found"
  cell, not answered by the assistant): more messages in the evening before the move,
  more messages during the day after the move.

### Still open

- [ ] Student reads the 4.6.5 chart against both recorded predictions — the good-
      morning/sleep-well ritual (wake/bedtime bump) and the evening-vs-daytime shift —
      and fills in "What I found."
- [ ] Build the greeting-word `RegexFeature` category (EN/NL/FR) as the more direct test
      of the good-morning/sleep-well claim, and/or the combined greeting-word-hours
      version, if 4.6.5's chart doesn't settle it on its own.

## Session 2026-09-22 (cont'd) — 4.6.6, message/burst length before vs. after Benji, and the graded-assignment check

- **Checked `learn.pttrn.io` for the actual `distribution` theme requirements**
  (courseId `60T1sfHrKYaeLJHUKwZR`): graded deliverable is **"Graded visualisation 3: a
  distribution before and after"** (les4), due **2026-10-05 22:00**, not yet handed in,
  all 5 criteria unchecked. Theme focus (quoted exactly, this is what "welke familie
  past" means): *"Een verdeling is een uitspraak over het proces erachter: welke familie
  past, en waarom hoort dat mechanisme bij jouw chat? Een histogram zonder die uitleg is
  een plaatje, geen verhaal."* — a fitted family AND a stated generating mechanism are
  both required, not just a shape. Also flagged: `verhaallijn`'s knockout example of a
  *bad* story is literally "mensen slapen 's nachts" — the same reasoning used to justify
  excluding night hours in 4.6.5; fine as a cleaning step, not usable as the finding
  itself.
- **New hypothesis from the student**: since Benji's birth (`BABY_DATE = 2025-05-11`,
  a different split than `MOVE_DATE`), she sends longer messages *and* more bursts than
  Louis, with a bigger gap after the baby than before. **Burst defined, her choice**:
  consecutive same-author messages before the other replies (run-length on `author`),
  one observation per burst, not per message.
- **Section 4.6.6 built — first real fitting in this notebook** (steps 5-9 of
  `goad_distribution_method`, not just shape-looking): `DistributionFitter` per
  (author × baby_period) group, both variables, plus `ECDFPlot` overlays labelled with
  the winning family and mean. All 8 fits succeeded, no `StopIteration` this time.
  **Results**: message length — `lognorm` won all 4 groups (mean chars: Roos 62.0→56.7,
  Louis 53.4→40.8; gap widened 8.6→15.9). Burst length — `nbinom` won 3 of 4 groups;
  **Louis-after-baby came back `binomial` instead**, flagged in the notebook as needing
  its own mechanism explanation rather than force-fitting the `nbinom` story (mean
  bursts: Roos 2.73→3.13, Louis 2.30→2.06; gap widened 0.43→1.07). Both variables'
  before/after gaps widened in the direction the student predicted — reported as numbers
  only, not asserted as a confirmed finding; interpretation left to her.
  Mechanism explanations drafted in the notebook (lognorm ~ product of factors, matching
  04.2's taxi-fare argument; nbinom ~ a continuation rate that itself varies) but framed
  conditionally ("if X wins, then..."), for the student to confirm or correct against
  what actually won. Re-executed clean via `jupyter nbconvert --execute --inplace`.
- **Not yet done, needed for the `spurious-correlations` rubric criterion**:
  `NullDistribution` (04.4 §4.4.3) on the before/after gap for one or both variables, to
  show the widened gap is more than shuffled-label noise — flagged in the notebook as
  the next step before this goes in the report.

### Still open

- [ ] Student reads 4.6.6 against her hypothesis and fills in "What I found," including
      the mechanism explanation for whichever family actually won (especially the
      Louis-after-baby `binomial` exception).
- [ ] Build `NullDistribution` check(s) for message length and/or burst length, before
      vs. after Benji — required for a strong `spurious-correlations` score.
- [ ] Decide which of this notebook's charts (response time, hour-of-day, message/burst
      length) becomes the actual graded submission for "Graded visualisation 3", due
      2026-10-05 — likely message length or burst length, since those are the ones with
      an actual family fit + mechanism behind them so far.
- [ ] Write the `reflectie-tekst` accompanying text for whichever chart is chosen,
      explicitly naming preattentive processing / text integration / storyline /
      spurious-correlations per the rubric's own wording.

## Session 2026-09-22 (cont'd) — 4.6.7, NullDistribution significance checks

- **Student asked for `NullDistribution` checks on all four charts built so far**
  (response time, hour-of-day, message length, burst length), plus asked to have the
  method explained — flagged she's "not yet an expert in deciphering these charts" and
  wants to understand the story before going further.
- **Statistic design, same shape for the three author-split variables**: a
  difference-in-differences / "gap of gaps" — `(Roos−Louis) after − (Roos−Louis) before`
  — rather than a plain before/after mean difference, since the student's hypotheses are
  about the *gap between people* changing, not just one variable moving. Label shuffled:
  `period`/`baby_period` (the before/after split), not `author`. Hour-of-day has no
  author split, so its check is simpler: evening share (hours 18:00-23:59, rounded from
  the 17:30 boundary), after minus before.
- **Results** (2,000 shuffles each, seed=4):
  - Response time gap-of-gaps: observed +0.27 min, **p=0.734 — not distinguishable from
    chance.** The "there might be a switch" hypothesis does not survive this check as
    stated (a gap-of-gaps test); doesn't rule out a plainer before/after shift for either
    person alone, which wasn't what was tested.
  - Evening-share shift: observed −0.265 (26.5 percentage points less evening after the
    move), **p=0.001 — solid.**
  - Message-length gap-of-gaps: observed +8.09 chars, **p=0.001 — solid.**
  - Burst-length gap-of-gaps: observed +0.63 messages/burst, **p=0.001 — solid.**
  - p=0.001 is the floor 2,000 shuffles can resolve (2/2001) for all three — reported as
    "p < 0.001", not treated as an exact number.
  Re-executed clean via `jupyter nbconvert --execute --inplace`.
- **Not yet interpreted with the student** — numbers reported factually in chat, "What I
  found" cell left blank in the notebook, same convention as every other prediction cell.

### Still open

- [ ] Student reads 4.6.7 against her hypotheses and fills in "What I found" — in
      particular, that the response-time "switch" hypothesis did not survive the
      gap-of-gaps check, while message length, burst length and evening share all did.
- [ ] Given the above, message length or burst length (both p=0.001, both with a fitted
      family already) are the strongest candidates for "Graded visualisation 3."

## Session 2026-09-22 (cont'd) — found the actual goal 4.17 exercise text; corrected `roos_04.4`

- **Checked the exact exercise description for "Graded visualisation 3" via
  `learn.pttrn.io`** (`get_course`, goal `4.17`), since the student asked how to improve
  the earlier work to match requirements. Exact text: *"de eigen chat splitsen op een
  bekend event (uit 03.3), berichten per dag en gaps binnen bursts aan beide kanten
  fitten, en het resultaat opschrijven als zin met parameters erin"* — split on a known
  event **from 03.3 specifically**, and fit **messages per day** and **gaps within
  bursts** (not message length or burst-message-count, which is what had been built in
  `roos_04.6` so far — those are good practice but don't match this specific ask).
  Also found: an earlier **draft deadline 2026-09-27** (before the final 2026-10-05
  one), and a linked reading (Barabási, *The origin of bursts and heavy tails in human
  dynamics*) recommended specifically as "the model of what your 'gaps within bursts'
  plot is showing" — useful for the required mechanism sentence.
  **Bug found**: `roos_04.4`'s `my_event` was `"2024-11-11"` (the NL move) — not
  actually one of 03.3's two named events (`"Positive pregnancy test"` 2024-09-01,
  `"Benji is born"` 2025-05-11), so it didn't satisfy "a known event from 03.3" as
  written.
- **Student chose "Benji is born"** (2025-05-11), matching the event the rest of her
  lesson-4 work (`roos_04.6`) is already built around. Fixed `my_event` in `roos_04.4`
  and re-executed the whole notebook clean.
- **Results, the two required variables, both already had working fit cells in
  `roos_04.4` §4.4.4** — needed only the corrected event:
  - **Messages per day**: before, `nbinom` mean 73.6 sd 74.1 (638 days); after, `nbinom`
    mean 33.9 sd 34.4 (486 days). `NullDistribution` on the mean difference: observed
    −39.8, p≈0.001 — solid, not shuffle-noise.
  - **Gaps within bursts** (exponential, gaps <1h): before, mean 223s (43,958 gaps);
    after, mean 376s (14,634 gaps). No `NullDistribution` check run on this one yet.

### Still open

- [ ] Write the required "one sentence with parameters" for both fits (goal 4.17) —
      draft given to the student in chat, hers to finalize in her own words.
- [ ] Run a `NullDistribution` check on the gaps-within-bursts scale shift (223s→376s),
      to match the "fit of distribution parameters where meaningful" the
      spurious-correlations row asks for on *both* required variables, not just one.
- [ ] Read the Barabási reading before writing the mechanism sentence for "gaps within
      bursts" — it's specifically about why this variable is heavy-tailed rather than a
      steady rate.
- [ ] Decide whether `roos_04.6`'s existing work (response time, hour-of-day, message
      length, burst-message-count) becomes supporting/exploratory material in the report,
      given `roos_04.4`'s messages-per-day + gaps-within-bursts fits are what goal 4.17
      actually asks for.
- [ ] Draft due 2026-09-27, final 2026-10-05 — both approaching.

## Session 2026-09-22 (cont'd) — burst-length deep dive, `roos_04.6`

- **Bug found and fixed, independent of the assistant's edits**: `roos_04.4`'s
  `my_event` had been changed to `"2025-11-05"` (day/month swapped from the intended
  `"2025-05-11"`), most likely from the notebook being open and re-saved in VS Code —
  exactly the sync risk flagged earlier this session. Student confirmed it was a
  mistake; fixed back to `"2025-05-11"` and re-executed clean, numbers match what was
  reported.
- **Documented the burst computation itself** (`roos_04.6`, new markdown after the
  burst-building cell): sort by timestamp, mark author-changes, `.cumsum()` into
  `run_id`, `.groupby(run_id)` collapses each run to one row with `burst_length` = run
  size. Includes a worked 8-message toy example.
- **New section 4.6.8**: two-panel small-multiples comparison (one panel per author,
  `sharex`/`sharey`), each panel overlaying the already-fitted before/after `pmf` curves
  (no raw histogram bars — reduces clutter, the fit was already verified in 4.6.6).
  Built specifically because the four-line ECDF was hard for the student to read;
  isolates the two comparisons (person, period) into panels instead of one shared axis.
- **Title, student's own choice**: "Mom takes the lead". Alternatives she proposed
  ("Mom tries to dominate", "Mom overexplains") were flagged in chat, not the notebook,
  as carrying an editorial judgement the burst-count data alone doesn't establish (more
  bursts could mean many things besides dominating a conversation) — her call to make,
  noted for the `verhaallijn`/tone discussion if it comes up at grading.
  Re-executed clean via `jupyter nbconvert --execute --inplace`.

### Still open

- [ ] Student picks a final title from the options discussed (or her own), with the
      tone trade-off in mind for live grading.
- [ ] Everything else from the previous entries still stands (NullDistribution on
      gaps-within-bursts, Barabási reading, reflectie-tekst, draft due 2026-09-27).

## Session 2026-09-22 (cont'd) — the actual finding is direction, not magnitude

- **Student corrected the framing**: the before/after difference in burst length isn't
  big in absolute terms for either person — the real finding is that Roos's and Louis's
  changes point in *opposite directions* (hers up, his down), which neither the ECDF
  (4.6.6) nor the side-by-side panels (4.6.8) actually put on the page directly.
- **New `DistributionDiffPlot` class added to `scripts/plots.py`**: draws a
  precomputed `(x, diff)` line with the area shaded to zero — takes the difference
  between two fitted densities/masses at the same x, rather than drawing the two
  distributions themselves. Same idea as 03.1's `SubtractBaseline` residual-over-time
  move, applied to two fitted distributions instead of two time windows.
- **New section 4.6.9**: one `after.pmf(x) − before.pmf(x)` line per person (Roos,
  Louis) on one shared axis with a `HorizontalLine` at zero, x-range set from the 99th
  percentile of every fitted distribution so neither curve gets cut off. Title kept as
  "Mom takes the lead" (student's choice, unchanged from the previous session).
  Re-executed clean via `jupyter nbconvert --execute --inplace`.

### Still open

- [ ] Student reads 4.6.9 against her own framing (opposite directions, not magnitude)
      and confirms it reads the way she intended.

## Session 2026-09-23 — nausea/fatigue vs. the pregnancy window, and a real bug hunt

- **New section 4.6.10**: nausea, fatigue, and both combined, each as a weekly count
  split "during pregnancy" vs. "the rest of the chat" (`PREGNANCY_START` to
  `BABY_DATE`, same window as `roos_03.3`), fitted with `DistributionFitter` and drawn
  with the same `DistributionDiffPlot` from 4.6.9 — reused, not rebuilt, per the
  student's "use the classes from the folder script" instruction. Nausea additionally
  got a `BarPlot` comparison chart (student asked for "a comparison and distribution"
  for that one specifically).
- **Bug found and fixed, twice**: `weekly_in_out`'s reindex step silently zeroed out
  every real count.
  1. First cause: `pd.date_range(min, max, freq="W")` built off the raw (time-of-day-
     carrying) timestamps never lines up with `resample("W")`'s midnight-Sunday bins —
     fixed by deriving the full week range from `resample` itself instead of
     reconstructing it by hand.
  2. That fix alone wasn't enough: `reset_index()` names the resulting column after
     the Series' own index name (`"timestamp_naive"`, inherited from upstream), not
     the generic `"index"` — so `.rename(columns={"index": "week"})` was a silent
     no-op, and `weekly["week"]` should have raised `KeyError` every time. Fixed with
     `.rename_axis("week")` before `reset_index()`, which names the column directly
     regardless of what it was called before.
  - **Process note, for next time**: the `KeyError` above did not show up in two
    consecutive background-executed verification passes — grepping the notebook
    immediately after a "completed" background-task notification returned stale,
    pre-edit output both times, and only running `jupyter nbconvert --execute` in the
    foreground (blocking, no `run_in_background`) surfaced the real error. Prefer
    synchronous execution when verifying a fix, especially after a result looked
    suspicious (uniform `mean 0.00 sd 0.00` across three unrelated categories was the
    tell).
- **Results, after the real fix** (weekly counts, `nbinom` throughout):
  - Nausea: outside pregnancy mean 0.03/week (sd 0.19); during, mean 0.81/week
    (sd 1.21) — roughly a 27x jump, n=120 weeks outside vs. 42 during.
  - Fatigue: outside mean 3.36/week (sd 3.00); during mean 4.31/week (sd 4.78) — a
    real but much smaller relative increase, consistent with "fatigue" being a far
    more generic word than "nausea" (already flagged in an earlier session).
  - Combined (either): outside mean 3.39/week (sd 3.02); during mean 5.05/week
    (sd 5.53).
  Re-executed clean via `jupyter nbconvert --execute --inplace` (foreground).

### Still open

- [ ] Student reads all three 4.6.10 charts — nausea should show the pregnancy window
      standing out sharply, fatigue more mildly, combined in between.

## Session 2026-09-28 — from notebook to a runnable pipeline (coding-grade prep)

- **New `scripts/chat_pipeline.py`**: load -> one goad `Pipeline` -> aggregate -> plot ->
  save, for the two final figures only ("Child vs. climbing: a trade in the
  conversation" and "Praat Nederlands met mij"). Run with
  `uv run python scripts/chat_pipeline.py`; writes `img/final_child_vs_climbing.png` and
  `img/final_praat_nederlands_met_mij.png`. Output checked against the notebook versions
  (same 47/7/46 → 10/22/68 language split).
- **Louis merge is now a pipeline step (`CanonicalAuthor`)**, first in the pipeline.
  Changed from the notebook's substring rule (`"louis" in name`) to an explicit alias
  map in `config.toml` under `[author_aliases]` (gitignored, so no names in git):
  `~Louis Chessé` and `Louis Ancien` → `Louis`. 18,386 messages merged; authors now
  `['Louis', 'Roos']`.
- Only `child` and `climbing` regexes moved to the script; the other six categories stay
  in 01.4.
- Setup verified: lefthook 2.1.12 installed and wired into `.git/hooks/pre-commit`;
  `.lefthook.yml` has **7** commands, not 5 (format, check, ty, clean-jupyter,
  notebooktester, lychee, pipeline-drift).

### Still open

- [ ] Student draws the flowchart of `chat_pipeline.py` and can explain each "what
      could break" point herself.
- [ ] Decide whether 01.4 should import `CanonicalAuthor` from the script instead of
      its own `canonical_author` function (two versions of one rule otherwise).

## Session 2026-09-28 (cont'd) — message length per topic: ridgeline + violin, `roos_04.6` §4.6.11

- **Hypothesis (Roos):** `planning_logistics` messages are longer than those of the other
  categories.
- **New classes in `scripts/plots.py`:** `RidgelinePlot` (KDE per group on log10 values,
  equal peak heights, median tick) and `ViolinPlot` (seaborn violin, `cut=0`,
  `density_norm="width"`, quartiles inside). Both grey-first via `highlight_groups=`.
  The param is deliberately *not* called `highlight`: that's a `PlotSettings` field, and
  `BasePlot.plot()` swallows any kwarg named after a settings field (found the hard
  way, planning_logistics rendered grey on the first run).
- **Categories:** the seven 01.4 topics (love, financial, child, climbing, running, food,
  planning_logistics), patterns copied unchanged. `food` is *commented out* in 01.4 with
  no logged reason; re-enabled here on request. `be_careful` left out.
- **Choices made by Claude, flagged in the notebook for Roos to accept or overrule:**
  multi-topic messages count in every topic they match (~13% of categorised messages);
  an `all messages` grey reference row; log x-axis; rows sorted by median.
- Medians printed (chars): food 108, financial 97, climbing 83, child 76, running 67,
  planning_logistics 66, love 24, all messages 32. Interpretation left to Roos.

### Still open

- [ ] Roos writes "What I found" under 4.6.11 against her prediction.
- [ ] Decide on the keyword-length confound (longer messages have more chances to match
      *any* regex, so every topic sits above `all messages`). Possible check: length
      matched against the number of regex hits, or a per-word-count baseline.
- [ ] Decide whether `food` stays (and log why it was commented out in 01.4).
- [ ] If the claim goes in the report: a `NullDistribution` shuffle of the topic label,
      same as 4.6.7.

## Session 2026-09-29 — pre-commit hook failure on `RidgelinePlot` (`ty`)

- **Symptom:** commit blocked by lefthook. The only failing check was `ty` at
  `scripts/plots.py`: `kde.set_bandwidth(kde.factor * bw_adjust)` → "Unsupported `*`
  operation". scipy's type hints give `kde.factor` a loose union (incl. `bytes`,
  `memoryview`), so `ty` can't prove the multiply is safe. Runtime was never wrong.
- **Fix (Roos):** set the bandwidth at construction instead,
  `stats.gaussian_kde(values, bw_method=lambda k: k.scotts_factor() * bw_adjust)`.
  Same result (`factor` defaults to Scott's factor), one line instead of two. Chosen over
  a `# ty: ignore`, which would hide the check rather than clarify the code.
- **Workflow gotcha:** after the edit the hook still failed and the file seemed to
  "revert". It was unstaged (`MM` in `git status`): lefthook checks the *staged* version
  and hides unstaged edits while it runs. Fix: `git add` the file, then commit.
- Committed as `19c4f42` + `865ab3a` (removed a stray `#` after `class ViolinPlot`).

### Still open

- [ ] `notebooktester` reported 9 failed / 4 timed out notebooks but doesn't block the
      commit (exit 0). Check whether `roos_04.6-distribution.ipynb` is among the
      failures; `07.1-social_graphs` fails on a string `timestamp` column in the own-chat
      loader.

## Session 2026-10-02 — reading `roos_04.3-penguins` (§4.3.1–§4.3.4), explained in Dutch

No code changed; a reading/coaching session. Claude explained each section, then Roos
answered a check question in her own words. Her answers, with the corrections they needed:

- **§4.3.1 (pooled fit fails):** "all families rejected *and* the two winners disagree" is
  more informative than "only `norm` rejected". It points at the grouping, not the family.
  Correction taken on board: the fit only *signals* the problem. *Which* groups to split
  by comes from the dataset or domain knowledge, not from the number of humps (two humps,
  three species: Adelie and Chinstrap merge into one).
- **§4.3.2 (per species):** ~100 birds per species can't tell normal / Weibull /
  skew-normal apart. "Pick the family the mechanism predicts" was unclear. Explained as:
  each family comes with a story about how the numbers are generated (sum → normal,
  product → lognormal, …). When the data doesn't choose between families, keep the one you
  can defend.
- **If `norm` were clearly rejected per species (Roos):** the grouping might still be
  wrong, e.g. split Gentoo by sex. Added: the other option is that the mechanism story
  itself is wrong.
- **§4.3.3 (Gentoo by sex):** why the fit couldn't see that mixture. Roos said it was the
  amount of data, and that the sexes are closer together than the species are. The second
  part is the key, made more precise: what counts is the distance *relative to the spread*
  (< ~2 sd merges into one bell). More data does not create a trough, so data volume is at
  most a side factor.
- **§4.3.4 (outliers relative to which population):** explained. Check question still
  unanswered (see below).

### Still open

- [ ] Roos writes her own answers under §4.3.1 and as reflection question 2 in
      `roos_04.3`.
- [ ] Answer the §4.3.4 check: why is a *missed* outlier (the 202 mm Adelie female, ordinary
      when pooled) worse than an *exaggerated* one? And which message or day in the chat
      would look ordinary for both authors together but unusual for one of them?
- [ ] §4.3.5 your-turn (pooled vs. per author on own chat), which overlaps with the
      per-author tip for 4.6.11.
- [ ] §4.4.4 of `roos_04.4`: step 1 (write down the expected effect of the event
      *before* running) is still to do.

### §4.4.4 event changed: Benji's birth → positive pregnancy test

- Roos edited the teacher's `04.4` directly and set `my_event = "2024-08-29"`
  (positive pregnancy test). That cell was copied into `roos_04.4`, replacing
  `2025-05-11` (Benji's birth), and `04.4` was restored with `git checkout`. The outputs
  from her run were not copied, so `roos_04.4` needs a re-run.
- [ ] Log *why* this event was chosen over Benji's birth. Note that "before" now covers
      only the dating period (before moving in, Nov 2024), but "after" mixes pregnancy,
      moving in and Benji.

## Session 2026-10-02 (cont'd) — reading `roos_04.5-fitting-the-residual`, explained in Dutch

No code changed. Claude explained the notebook's core ideas in Dutch: fit a model, subtract
it, and fit a distribution to the residual to decide whether the model is done (noise →
stop; structure over time → missing mechanism, and *when* it starts is the finding). Covered
the COVID example (line on pre-vaccination window: residual is noise, six families tie;
full window: residual has a shape, every family rejected, points at vaccination) and §4.5.5
(outliers via tail probability under the fitted family, not the z-score rule: 0 vs. 2 days
flagged).

### Still open

- [ ] Roos answers the check question from this session in her own words.
- [ ] Reflection question 4 of `roos_04.5` links to the graded assignment: flag unusual
      days in `roos_04.4`'s messages-per-day fit (after the re-run with the new event) and
      explain them from her own knowledge of the chat.

## Session 2026-10-02 (cont'd) — Graded visualisation 3 built: `roos_04.6` §4.6.12

Built from Roos's own detailed brief (event, windows, families, keyword lists, plots,
robustness checks all specified by her). Event = positive test **2024-08-29**, 77 days each
side (13 Jun – 28 Aug / 29 Aug – 13 Nov 2024, both long-distance).

- **Followed the notebooks over the brief where they conflicted:** burst threshold 60 min
  (from `roos_04.4`), not 30; event date from `roos_04.4`. **Unresolved:** `roos_03.3` puts
  the positive test on 2024-09-01, and the assignment asks for "an event from 03.3".
- **Keyword lists:** the brief's new, stricter FR/NL/EN lists (accent-stripped, word
  boundaries), not §4.6.10's patterns (which match "crevettes").
- **Found:** goad's `DistributionFitter` fits `nbinom` through `scipy.stats.fit`, which treats
  r as an integer (r came back 1.00 / 2.00). Count fits now done by direct MLE in the notebook.
- **New `BeforeAfterDensityPlot` in `scripts/plots.py`**: overlaid density histograms on
  identical bins, fitted curve per side, direct labels; log-x mode converts curves to
  density per log10 so bars and curves match. Figures saved to `figures/vis3_*.png/svg`.
- **Results:** messages/day NB μ 85.2 → 95.5, r 1.38 → 2.02 (Poisson ΔAIC ≈ 2,400); shuffle
  p = 0.25. Gaps: truncated lognormal best of three on both sides (e^μ 39 s, σ 1.88 → 1.80),
  exponential ΔAIC ≈ 10,000; power-law tail α 1.10 / 1.21 above ~2 min, tied with lognormal
  on the tail. Fatigue-or-nausea 0.52 → 2.01 messages/day (nausea 0 → 0.45), p ≈ 0.001.
- **Weakens the story:** the daily-volume result is window-sensitive (μ +58% at 4 weeks,
  +37% at 8, +12% at 11; p ≈ 0.009 / 0.008 / 0.25), driven by very quiet weeks ending 14 and
  21 Aug 2024. Gap tempo is robust across 15/30/60-min thresholds (median change ≤ 5%).
- **Reference values reproduced** at 30 min (e^μ 33 s, σ 1.71/1.64) and for symptom rates;
  messages/day μ is ~10% below the earlier parse (85/96 vs 97/103), cause unknown.

### Still open

- [ ] Roos: were we together in mid-August 2024? Decides how to explain the quiet weeks.
- [ ] Align the event date between `roos_03.3` (2024-09-01) and `roos_04.4`/§4.6.12 (2024-08-29).
- [ ] Fifth rubric criterion (from the course page), then the reflectie-tekst.
- [ ] Before submitting: clear the keyword-snippet cell's output, and cell 3's `own.head()`
      output (shows full message texts).
- **Answered (Roos, 2026-10-02):** event date 2024-08-29 verified in her data (so `roos_03.3`'s
  2024-09-01 marker is the one that's off). We were together mid-August 2024, which explains
  the quiet weeks. Messages per day matters less to her than gaps and content.
- **Fifth rubric criterion: not findable in the repo.** `checklist.md` and
  `references/leerdoelen/les4.md` hold no rubric; only teacher-feedback slugs appear in the log
  (`preattentive-gestalt`, `verhaallijn`, `spurious-correlations`; `reflectie-tekst` likewise
  written as a slug). The `learn` server timed out this session. Roos to check the course page.

## Session 2026-10-02 (cont'd) — §4.6.12 revised, new §4.6.13 "the hard weeks"

- **§4.6.12 revised** with Roos's answers: event date 2024-08-29 verified (`roos_03.3`'s
  2024-09-01 marker is the one to fix; it is the start of pregnancy week 5); messages per day
  relabelled a required side layer; the mid-August dip explained as a visit (together, chat
  quiet); result sentence updated to say so.
- **New §4.6.13**, Roos's focus, story "How to prepare for a next pregnancy". Hard weeks =
  pregnancy weeks 5–14 (`PREGNANCY_START` + 5 to + 15 weeks = 1 Sep – 9 Nov 2024, 70 days),
  03.3's week convention. Baseline = long-distance chat before the test (383 days; includes
  pregnancy weeks 0–4, flagged as conservative). Rates per 1,000 messages.
- **Results:** nausea-or-fatigue 8.2 → 22.6 per 1,000 (×2.7, shuffle p ≈ 0.001); nausea 0.06 →
  5.17 per 1,000, almost only in the hard weeks. Weekly: normal weeks NB μ 4.9, r 5.26
  (ΔAIC ≈ 10 over Poisson); 6 of 10 hard weeks beyond the 99th percentile (14); worst week 12
  (30 messages); edges (weeks 5, 14) mildest. Later periods 7.3 / 10.0 per 1,000 (lower bounds:
  in person). Chat busier during hard weeks (96.6 vs 83.9 messages/day).
- Figures: `figures/trimester_timeline`, `trimester_standout`, `trimester_periods`.
  `BeforeAfterDensityPlot` now draws separate bars for discrete data.

### Still open

- [ ] `MOVE_DATE` in the notebook is 2024-11-01, but Roos remembers mid-November. If the real
      date is later, the hard weeks were fully long-distance. Confirm and correct.
- [ ] Roos writes "My lessons for a next pregnancy" at the end of §4.6.13.
- [ ] Fifth rubric criterion, then the reflectie-tekst.

## Session 2026-10-02 (cont'd) — choosing the hand-in chart; testing it

- **Before/during/after dot chart** added to §4.6.13, after the stand-out chart
  (`DotStripPlot`, new in `scripts/plots.py`; `figures/trimester_before_during_after`). One dot
  per week, per 1,000 messages: before 8.2 (54 weeks), hard weeks 22.6 (10), weeks 15–40 until
  Benji 7.3 (26); shuffle p ≈ 0.001 for both comparisons. Only 6 of 54 before and 1 of 26 after
  are above the mildest hard week. The "after" column is a lower bound (living together).
  "After" = rest of the pregnancy, Claude's reading of "the period after"; after-Benji
  (10.0 per 1,000) could be added as a fourth column.
- **Chart types discussed:** the dot chart is a jittered strip plot. It shows the spread of the
  weeks but fits no family, so it is a comparison chart, not a distribution chart in the
  lesson-4 sense. The stand-out chart is one (fitted NB + tail probabilities).
- **Roos's decision: hand in the stand-out chart** ("6 of the 10 hard weeks…") for feedback and
  improve it from there. Caveat discussed: it doesn't follow goal 4.17's wording literally
  (messages per day and gaps within bursts, fitted on both sides). Those fits are in §4.6.12
  (Step 1, Step 2, parameter table, result sentence); point the grader there in the
  submission text.
- **Stand-out chart updated:** title "How to survive a pregnancy: prepare and plan.", subtitle
  "6 of the 10 hard weeks would be a 1-in-100 week in normal times". μ and r explained on the
  chart: μ = the average symptom messages per normal week (4.9); r = how much that weekly
  rate wanders (5.26; smaller r = more variation, r → ∞ = Poisson). The NB is explained as a
  gamma-Poisson mixture: each week its own rate, messages at random moments within it.
- **Tests added before the chart** (new cell):
  1. Parametric bootstrap: the NB fits the 54 normal weeks (simulated samples show a gap
     this large in 65% of cases).
  2. 0 of 54 normal weeks are above the 99% line (14); the family expects 0.4.
  3. P(≥ 6 of 10 normal weeks beyond the line) = 3.4e-11, assuming independent weeks (an
     order of magnitude, since neighbouring weeks are not fully independent).
  4. NB on the 10 hard weeks: μ 15.3 (90% range 12.5–18.5), but r from 7.1 to unbounded and
     ΔAIC 1.2 against Poisson.
- **Roos's footnote checked and sharpened:** "no family fitted to the 10 hard weeks, too little
  data" is correct for the *shape*, not the *average*. The footnote now reads: "No family
  fitted to the 10 hard weeks: too few to tell a negative binomial from a Poisson. Their
  average is clear: 15.3 a week (90% range 12.5-18.5)."

### Still open

- [ ] Cosmetic: the grey label on the stand-out chart touches the dotted 99% line.
- [ ] Submission text: name §4.6.12 as where goal 4.17's required fits and sentence are.

## 2026-10-06: is the stand-out chart a distribution?

- **Question (Roos):** the y-axis of `distribution_3_survive_pregnancy_prepare_plan.png` says
  "share of normal weeks", not "probability". Is it a distribution?
- **Answer:** yes. `BeforeAfterDensityPlot(..., discrete=True)` with one bin per count uses
  `density=True`, so each grey bar is the share of the 54 normal weeks with exactly that count
  (bars sum to 1), i.e. an empirical P(week = k); the line is the NB pmf on the same scale. The
  crimson dots are *not* on that scale: their height is only stacking.
- **Added in roos_04.6 (after the stand-out chart):** a second version with both groups as
  probability distributions (grey = normal weeks + NB pmf, crimson = the 10 hard weeks, each
  0.1), y-label "probability: share of weeks with exactly this many". Saved as
  `figures/trimester_standout_pmf.png`. Checked: both sets of bars sum to 1.00.

### Still open

- [ ] Roos: decide which of the two versions goes in the report, and why.

## 2026-10-06 (cont'd): roos_04.6 split into four notebooks

`roos_04.6-distribution.ipynb` was too long (99 cells). Split into four notebooks. The
§-numbers stay the same, and each notebook links to the other three:

- `roos_04.6a-move-and-baby.ipynb`: §4.6.1–4.6.9 (response time, hour of day, length and
  bursts, null check, opposite directions).
- `roos_04.6b-symptoms-and-topics.ipynb`: §4.6.10–4.6.11.
- `roos_04.6c-graded-vis3-positive-test.ipynb`: §4.6.12 (graded visualisation 3).
- `roos_04.6d-hard-weeks.ipynb`: §4.6.13.

Shared code is in `notebooks/lesson4/roos_046_shared.py`: `load_own()`, the four dates, `SIDE_COLORS`,
`save()` and `local_chat()`, plus *copies* of `normalise`, `SYMPTOM_PATTERNS` and `fit_daily` for
4.6d. Roos chose this so the explained versions stay visible in 4.6c. **Change both together.**
Checked: the copies match 4.6c, each notebook runs on its own, and all 39 moved cells give
the same output as the original.

### Still open

- [ ] Roos: delete the old `roos_04.6-distribution.ipynb` once she has checked the four new
      ones. It is still in git history (commit 4b01da6).

## 2026-10-06 (cont'd): is the hard-weeks difference real? (4.6d, after the stand-out chart)

Added a label-shuffle check (`NullDistribution`, 10,000 shuffles over the 64 weeks) plus a
visual, `figures/trimester_standout_shuffle.png`:

1. Gap in averages: +10.4 symptom messages a week (bootstrap 95% range +6.9 to +14.2); 0 of
   10,000 shuffles reach it, so p ≤ 0.0001.
2. 6 of 10 beyond the 99% line: 0 of 10,000 shuffles; exact hypergeometric chance 2.8e-6.
3. Order kept: the 45 runs of 10 consecutive normal weeks average 3.7–7.5, against 15.3 for the
   hard weeks; 0 of 45 reach it (p ≤ 1/46).
- Overlap: a random hard week beats a random normal week in 98% of pairs; 4 of 10 hard weeks
  fall inside the normal range (8, 10, 11, 13).
- Explained to Roos: p is *not* "the chance something is going on".

### Still open

- [ ] Roos: decide whether to model it as a rate (NB regression with log(messages) as offset, rate
      ratio + CI) instead of / next to weekly counts. Not built; her call.

## 2026-10-06 (cont'd): stand-out chart with both groups fitted (4.6d)

New chart `figures/trimester_standout_both.png`, after the shuffle test: the stand-out chart's
title, subtitle and labels, but both groups as probability distributions, each with a fitted
negative binomial (normal μ 4.9, r 5.26; hard μ 15.3, r 16.1). The shuffle result is in the
footnote. Second footnote: the hard-weeks curve is for comparison only (10 weeks, ΔAIC 1.2 against
Poisson); the average is what is solid.

### Still open

- [ ] Roos: choose between `trimester_standout`, `_pmf` and `_both` for the report.
- Subtitle of `trimester_standout_both` (Roos's choice, option 1 of 4): "In the hard weeks, nausea and
  fatigue came up three times as often: 15 messages a week instead of 5". "Three times" is by weekly
  counts (15.3 / 4.9); per 1,000 messages it is 2.7×. An assert in the cell guards the written-out
  "three". Footnote now rounds to "about 1 in 360,000".
- `trimester_standout_both` reworked (Roos's request): lines only (the two fitted NB pmfs, no bars),
  the "54 normal long-distance weeks…" text block removed, short direct labels at the peaks. Below
  the chart: which line is which (54 long-distance weeks, avg 4.9; 10 hard weeks = pregnancy weeks
  5-14, avg 15.3) and the shuffle result, opening with "Too large to be a coincidence." The
  "hard-weeks curve fitted for comparison only" footnote was dropped from the chart; that caveat
  now lives only in the markdown cell above it.
- `trimester_standout_both` trimmed further (Roos's request): 99% line and its text removed; crimson
  label is now just "hard weeks" (the "6 of 10 beyond…" claim needed the line); footnote no longer says
  "too large to be a coincidence" but "Chance alone does not explain this gap: …occurred 0 times
  (p ≤ 0.0001)"; the "6 of 10 beyond the line by chance / no run of 10 weeks" line removed.

## Session 2026-10-06 (cont'd) — `roos_pipeline/`: five figures as one goad pipeline (coding grade)

**Goal (Roos):** the five graded visualisations rebuilt by one pipeline in a folder of her own:
Praat Nederlands met mij (`roos_01.4` code cell 13 + `roos_02.5` code cell 5), Baby is taking over
(`roos_03.3` code cells 6 + 7), How to survive a pregnancy: prepare and plan (`roos_04.6d`).
Run: `uv run python -m roos_pipeline` (~13 s). Flowchart + explanations in `roos_pipeline/README.md`.

**Decisions (Roos's, asked one by one):**
- Folder `roos_pipeline/` (same `roos_` convention; *not* `dev/`, which is gitignored).
- Her own code moves out of `scripts/`: `chat_pipeline.py` absorbed and removed (still in
  `c3edc36`), her 7 plot classes moved to `roos_pipeline/plots.py` unchanged,
  `scripts/plots.py` restored to the teacher's version. Six roos_ notebooks now import
  them from `roos_pipeline.plots`; `roos_046_shared.py` imports dates, `normalise`,
  `SYMPTOM_PATTERNS`, `fit_daily` from the pipeline (one copy, no drift).
  `roos_04.6d` executed end to end afterwards: runs.
- **One date per event** (the notebooks disagreed): positive test **2024-08-29** (not 03.3's
  09-01), birth **2025-05-11** (not chat_pipeline's 05-01). All in `settings.py`.
- Topic regexes: the **03.3 versions** (not 01.4's narrower ones).
- Distribution figure: **`trimester_standout_both`** (closes the open choice of 2026-10-06).
- pytest added (`uv add --dev pytest`, uv.lock updated) + lefthook command
  `roos-pipeline-tests`, which runs only when `roos_pipeline/`, its tests, `pyproject.toml`
  or `uv.lock` change. Roos asked what pytest adds over ruff/ty: ruff = style, ty = types,
  pytest = *wrong numbers*; only looking catches a misleading chart.

**Structure:** 4 stages, each boundary a file in `data/processed/roos/` (gitignored):
prepare → `1_messages` → features (per message) → `2_features` → aggregate (grain changes;
NB fits + 10,000 shuffles) → `3_*` → figures (draw only). Any stage reruns from the
previous file (`--stage`, `--from`); a test deletes stages 1-2's output and redraws.

**Real bugs caught, and by what:**
- **Test → regex false positives (the main one).** Writing "surprise is not climbing" as a
  test made it fail. On the real chat, 186 of 1,764 climbing messages counted *only* through
  the 03.3 additions: surprise(d/s) 133, entreprise(s) 25, poffertjes 5. Child: `partou`
  matched *partout* (23), `couches?` matched *coucher* (47) and *accouchement*. Roos chose:
  `\bpof\b`, `\bprises?\b`, `partou\b`, `couches?\b` (accepting that *accouchement* no longer
  counts). Counts after: child 2,274 → 2,163, climbing 1,764 → 1,605, sports 2,752 → 2,595.
  Follow-up (Roos): `accouchements?` added to child as its own word, so childbirth still
  counts; child 2,163 → 2,207. Test added.
  Story of "Baby is taking over" unchanged by eye. The 03.3 notebook still has the old regex.
- **Test → my own edit.** Applying that fix through a shell heredoc wrote backspace characters
  instead of `\b`; two tests failed at once. Invisible by eye.
- **ty → pandas typing.** `groupby(day).size().rename(...)`: stubs say Series *or* DataFrame.
  Fixed with `day.value_counts()`, no `ty: ignore` (same choice as 2026-09-29).
- **Looking → layout.** Figure 1 repeated the y-label on the right panel; no test could see it.
  Fixed via `PlotSettings.subplot_ylabels`.
- **Hook verified:** `lefthook run pre-commit --command roos-pipeline-tests` failed (🥊) while
  the 4 regex tests failed, i.e. a commit would have been blocked. Now 33/33 pass.

Numbers reproduced exactly before the regex fix: language 47/7/46 → 10/22/68; normal weeks
μ 4.85, r 5.26; hard weeks μ 15.3, r 16.1; gap +10.4, 0 of 10,000 shuffles.

### Still open

- [ ] Roos: redraw/adjust the flowchart herself (README has a Mermaid draft) and explain each
      boundary in her own words: what breaks if merged, what reruns alone.
- [ ] Roos: commit (the hook runs the tests); `uv.lock` must be in the commit.
- [ ] Update `roos_03.3`'s regex to the fixed version, or note there that it differs.
- [ ] Figure 5's words "three times" are guarded: the pipeline stops if the ratio no longer
      rounds to 3.

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

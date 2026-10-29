# ytcs.github.io

Personal research site of To Chin Yu. Jekyll on GitHub Pages (built by
`.github/workflows/jekyll.yml`; Pages source is "GitHub Actions"), KaTeX via
jektex (server-side; macros in `_config.yml` under `jektex.macros`).

Structure: a wiki, not a blog. Pages are reference articles that are revised in
place and link to each other; none is a dated entry.
- `_data/wiki.yml` lists the sections of the home page. A page joins a section
  through its `category` front-matter field.
- Pages are `_notes/<topic>/<slug>.md`, served at `/<topic>/<slug>/`, ordered
  within a section by the `order` front-matter field. Front matter: `title`,
  `date` (creation; shown only as the revision date), `category`, `order`,
  `description` (plain text, no math: shown on the home page and used as the
  meta description; without it the first paragraph, with its math, ends up in
  the page head), optional `status` and `excludes` (plain text, shown in the
  facts table under the title), optional `last_updated`. No reference to where
  the result came from.
- Link pages to each other with ordinary Markdown links to their URLs; the
  layout lists the pages that link to the current one at the bottom.
- A page is one result, or one map, written as lecture notes: full rigor,
  pedagogical order.
- Theorem-like blocks: `<div class="theorem" markdown="1">` (also `lemma`,
  `proposition`, `corollary`, `definition`, `remark`, `example`, `proof`), first
  paragraph starting `**Theorem 1** (Name).` / `*Proof.*`. Number equations with
  `\tag{n}` and refer to them as (n).
- Math: `$$…$$` inline inside a paragraph, display when it stands alone between
  blank lines.

Read this before touching the design or writing any prose for the site.

## Design

The site is for reading long arguments with equations. Every design decision
serves that. Nothing moves, floats or performs behind the text.

Do not use these. Each is a recognised marker of generated design (sources:
anthropics/skills `frontend-design`, pbakaus/impeccable, Leonxlnx/taste-skill,
Nutlope/hallmark):

- Blurred pastel blobs, aurora/mesh gradients, glows, grain, glassmorphism,
  backdrop blur.
- Inter, Roboto, Space Grotesk, Geist, Fraunces as the face of the site; the
  stock "Crimson/Playfair heading + Inter body" pairing.
- Warm cream ground + serif display + gold/terracotta accent as a "tasteful"
  default.
- Monospace as a costume ("technical" lowercase mono site titles). Monospace is
  for code only.
- Cards, rounded boxes with soft shadows, a card around the bio.
- Hover scale/lift on images or list rows; `transition: all`; fade-in on scroll.
- Floating widgets: auto-hiding control panels, toggle switches, edge tabs,
  music players.
- Unicode glyphs as icons (▶ ›), emoji, "→" appended to links, "·" or "•" meta
  strings, ALL-CAPS tracked eyebrow labels, numbered "01 / 02" section markers.

Do this instead:

- One column, measure 60–70 characters. Body text 18–20px, line-height ~1.55,
  left-aligned, full-strength ink (not grey).
- One serif text face for body and headings, chosen to sit well next to KaTeX
  (KaTeX is Computer Modern–like). Mono for code only. Self-host fonts.
- Plain near-white ground in light mode, near-black in dark mode. One
  link/accent colour. Dark mode follows `prefers-color-scheme`; a manual
  override is a plain text link in the header, applied before first paint.
- Headings: more space above than below, `text-wrap: balance`. Body
  `text-wrap: pretty`, `hanging-punctuation`, old-style figures in prose and
  tabular figures in tables.
- Figures sized to the column, never wider than the viewport, with a caption.
  Display math scrolls horizontally on narrow screens instead of overflowing.
- Table of contents only on long posts: in the right margin on wide screens,
  a `<details>` at the top on narrow screens. Never overlapping text.
- Footnotes as real footnotes (kramdown `[^1]`), references as a plain list.
- Home page: a first-person line or two about the author, then the wiki's
  sections, each with its pages (title + one-sentence summary). No tabs, no
  dates, no previous/next links.
- Visible keyboard focus, `prefers-reduced-motion` respected, readable with
  JS off, contrast ≥ 4.5:1. Check at 390px wide before calling anything done.

## Writing voice

Notes read as self-contained manuscripts in the register of a standard
academic paper or a good set of lecture notes.

- Use "we" for the reasoning and the computations. No first-person opinion,
  no asides about what the author believes, finds plausible, or would like to
  know next.
- Withhold subjective judgement. Where there are several possibilities (escape
  routes, interpretations, readings of a tension), state each with its
  conditions and leave the choice to the reader.
- Self-contained: never mention how the result was produced. No references to
  pipelines, campaigns, agents, LLMs, referees, scripts, code directories or
  ledgers. "Two independent calculations agree" is fine; "our pipeline" is
  not ("the setup of section 5" instead).
- Open with the question and the answer, stated with their conditions. Never a
  hook question, never "In this note we will explore".
- Credit prior work near the start, in prose, and say what is added.
- Hedges are specific ("this assumes γ is constant across caps"), never blanket.
- Vary sentence length. Plain words.
- Define terms inside sentences, not as bold labels in bullet lists. Bullets
  are for genuinely list-shaped things.
- Headings name what happens in the section ("Curvature and the BAO
  distances"), not grand abstractions ("The Fundamental Dynamic Equation").
- End where the argument ends: limitations and open questions, stated neutrally.
  No recap section, no "What is new" list at the end.
- Corrections after publication are dated inline: "Update (2026-01-24): …".

Words and patterns to remove on sight: crucial, fundamental, profound, powerful,
remarkable, elegant, delve, "the key insight", "it's not X, it's Y",
"isn't just", "the honest reading", "worth noticing", "the whole mechanism",
signposting that announces instead of saying ("Here are the assumptions"),
sentences that state the obvious or vouch for the work ("Each is a smaller
statement, proved and checked, that stands on its own"),
dense "not X but Y" constructions, rhetorical triads, "Let's dive in", "Have you ever", em dashes
used as punchline drumrolls (two or three per post is plenty), "mathematical
certainty", titles that claim more than the post shows ("X Is the Key to…",
"solves").

## Presenting mathematics

Common complaints about AI-written maths: proofs that are correct but give no
understanding (Maynard: "very difficult to really extract any human
understanding"), "not written for humans" (Gómez-Serrano), constructions that
appear from nowhere, prior results cited without saying why they apply
(Moshkovitz), redundant lemmas at the wrong level of abstraction (Tao), and
sign errors nobody checked. The rules below are the usual counter (Halmos "How
to write mathematics", Serre "How to write mathematics badly", Knuth
"Mathematical Writing", Tao's writing advice, Gowers).

1. State the question and the answer first, informally, in the first
   paragraph. The reader should know where they're going.
2. Motivate before formalising. Give the concrete example or the picture first,
   then the general statement (Gowers labels his "Example 1", "General
   discussion", "Example 1 continued").
3. Introduce each symbol where it's first needed, with the reason it's needed.
   No notation dumps up front. Fewer symbols is better; never reuse one for two
   things. If a symbol is used once, you probably don't need it.
4. Give the idea of an argument before its steps: "The point is that…",
   "The difficulty is…, and we get around it by…". Then the details.
5. Weight the steps honestly. Say which step is routine and which is the real
   content. Don't belabour algebra and then wave through the hard step with
   "it follows that".
6. Keep the dependency order linear. A lemma appears when it's used and you say
   why it's needed. No forward references to unmotivated objects.
7. Label the epistemic status of every claim: model assumption, heuristic,
   derivation under the model, empirical observation, or conjecture. Call
   something a theorem only if the post proves it, and say what it is a
   theorem *about*. A calculation inside a toy model is a calculation, not an
   "impossibility theorem".
8. Check it. Limiting cases, dimensions/units, a worked numerical example with
   real numbers, signs. Say what observation would falsify the model.
9. Every display equation is part of a sentence, punctuated, and followed by
   what it says in words. Display only what's referenced or central; number
   what's referenced.
10. Cite with a reason: what the cited result says and why it applies here.
11. Say what you don't know and what's open. Tao: "it is not yet fully
    understood whether…". Baez: "There may be a simpler argument, but this is
    what I've seen."

Before publishing a post, grep it for the words above, read the first
paragraph alone (does it state the question and the answer?), and read only
the display equations in order (does the chain make sense without the prose?
does the prose add the why?).

## Review before publishing a note

No note goes live until it has passed both stages below. Run the agents in
parallel and in the background. Each starts fresh and gets only the file paths
and its criteria, never the author's opinion of the note, so its read isn't
anchored. Agents are read-only and work in the scratchpad.

**1. Verification** (one agent per note). Treat every claim in the source as
unverified; source results may have been produced and checked by LLM agents only.
- Redo every proof and derivation by hand (sympy is fine).
- Recompute the headline numbers with the agent's *own* code where feasible.
  Re-run the source material's own code for anything too heavy to
  reimplement.
- Check every quoted measurement and every bibliography entry against the
  source (arXiv abs pages, the papers' equations and tables). Never fill in
  author initials or titles from memory.
- Report each item as VERIFIED / DISCREPANCY (with numbers) / NOT CHECKED (why),
  plus anything overstated or misleading.

**2. Red team** (three agents, each reading every note in the batch, one
dimension each). Target reader: a physics PhD student or postdoc outside the
note's subfield.
- *Readability and ease of following:* undefined or late-defined symbols and
  jargon, steps the reader can't reproduce, overloaded sentences, unclear
  tables or units, places where an example or one-line intuition is missing.
- *AI-slop and writing tells:* the "Writing voice" list above plus the
  reviewer's own knowledge of LLM prose: stock intensifiers, contrast
  templates, triads, signposting that announces instead of saying, bold run-in
  labels, uniform rhythm, stacked hedges, recap sentences, grand framing; also
  stilted over-formality, and any mention of how the result was produced
  (pipelines, agents, code). The target register is a standard academic paper.
  Verdict: how likely a skeptical physicist is to clock it as LLM-written.
- *Logical progression and linear build-up:* the note's actual dependency
  outline, forward references, objects used before they're motivated,
  hypotheses far from where they're used, epistemic labelling (proved /
  numerical / sketch / cited), weighting of routine against hard steps,
  opening states question and answer, ending stops with the argument; a better
  section order if one exists.
Each finding quotes the phrase (≤15 words), says what goes wrong, and gives a
concrete fix, ordered by severity.

**Audit trail.** Every note has a review log at
`_reviews/<project-id>/<slug>.md` (underscore folder: never published). Each
round appends a section: date, the reviewer brief in one line, every finding
(numbered, quoted phrase, severity), and against each finding the response:
FIXED (what changed), DECLINED (why), or DEFERRED (why). Author instructions
that settle a question (e.g. "academic tone, no subjective judgement") are
recorded there too, so later rounds don't reopen them.

**Then:** fix every verification discrepancy and every blocking or significant
red-team finding, or record in the log why not. After the revision, run a
second round: resume the same reviewer agents with SendMessage if they are
still reachable, otherwise start fresh ones. Either way give them the review
log and the revised file, and ask them to (a) check each of their previous
findings against the revision and mark it resolved or not, and (b) report only
new problems introduced by the revision. Don't let a later round relitigate a
finding the log marks DECLINED with a reason unless the reviewer shows the
reason is wrong. Repeat until no blocking or significant findings remain.
Build the site and check the rendered page (no KaTeX errors, theorem blocks,
tables, figures and phone width) before committing.

## Deploying

Pushing to `master` deploys via the GitHub Pages workflow. After pushing, do
not wait for or poll the deploy (watching `gh run` tends to hang); the author
checks the live page. Before pushing, verify the local build instead: the
rendered body *and* the generated `<head>` (meta tags must not contain
rendered math or HTML).

# ytcs.github.io

Personal blog of To Chin Yu: physics, maths, finance, ML. Jekyll on GitHub Pages
(built by `.github/workflows/jekyll.yml`), KaTeX via jektex. Posts in `_posts/`.

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
- Home page: a first-person line or two about the author, then posts by year.
  A multi-part series appears once, with its parts listed beneath it.
- Visible keyboard focus, `prefers-reduced-motion` respected, readable with
  JS off, contrast ≥ 4.5:1. Check at 390px wide before calling anything done.

## Writing voice

The reference for voice is the author's own 2023 posts
(`2023-09-18-stable-distribution-with-minimal-information.md`): first person,
says what they noticed, says what they believe and how sure they are ("I feel
there must be a deeper reason… I present here an informal argument"). Other
models: Tao (What's new), Gowers, Baez, Aaronson, Olah, Jeremy Kun.

- Open with the occasion or the claim. "Yesterday several people asked my
  opinion of a preprint…" (Aaronson). "Humans don't start their thinking from
  scratch every second." (Olah). Never a hook question, never "In this post we
  will explore".
- Write in first person. Say "I don't know", "I'm not sure this is right",
  "I was confused by this". Hedges should be specific ("this assumes γ is
  constant across caps"), not blanket.
- Vary sentence length. Plain words. Dry humour is fine when it's about the
  subject.
- Define terms inside sentences, not as bold labels in bullet lists. Bullets
  are for genuinely list-shaped things.
- Headings name what happens in the section ("Fitting γ to S&P data"), not
  grand abstractions ("The Fundamental Dynamic Equation"). Short essays can
  have no headings at all.
- End when the argument ends: an open question, a limitation, what would change
  your mind. No "Conclusion" section restating the post.
- Corrections are dated inline: "Update (2026-01-24): …".

Words and patterns to remove on sight (current posts use "fundamental" ×52,
"crucial" ×45, 74 bold-label bullets): crucial, fundamental, profound, powerful,
remarkable, elegant, delve, "the key insight", "it's not X, it's Y",
"isn't just", rhetorical triads, "Let's dive in", "Have you ever", em dashes
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

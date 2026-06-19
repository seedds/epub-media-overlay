# Sentence Segmentation Logic

This document explains how `mark_sentence.py` breaks chapter text into read-aloud
chunks ("segments"), with emphasis on the sentence-boundary detection that drives
it. It also records the rationale behind some non-obvious edge-case handling.

## Overview

The module wraps each segment of visible text in its own span so a media overlay
can highlight it during playback. There are four cooperating stages (see the
module docstring's "Section map"):

| Function | Answers |
| --- | --- |
| `_build_char_map()` | "If I flatten this DOM subtree into a string, which DOM node does each character position map to?" |
| `_get_sentence_aware_segment_boundaries()` | "Given that string, where are the best places to break it into read-aloud chunks?" |
| `_create_segment_spans()` | "Using those boundaries, how do I rebuild the subtree as segmented HTML without losing structure?" |
| `validate_text_consistency()` | "Did we preserve exactly what the user will see/read?" |

This doc focuses on the second stage: boundary detection.

## Two-level boundary detection

Boundary detection happens at two granularities and is then reconciled:

1. **Sentence boundaries** — `_get_sentence_boundaries()` (`mark_sentence.py:620`).
   Uses NLTK Punkt to find sentence spans across the whole text.
2. **Phrase boundaries within a sentence** — `_get_segment_boundaries_in_sentence()`
   (`mark_sentence.py:724`). Splits a single sentence into smaller read-aloud
   phrases (e.g. at commas, dashes, closing quotes).
3. **Orchestration** — `_get_sentence_aware_segment_boundaries()`
   (`mark_sentence.py:830`). Walks the sentence spans, splits each into phrases,
   merges fragments that are too short or punctuation-only into a neighbour, and
   **fills gaps** so the returned boundaries cover the full string with no holes.

The orchestrator guarantees full coverage: any gap between detected sentence
spans, and any trailing remainder after the last sentence, is preserved as its
own boundary. This matters because the char map relies on every character being
accounted for.

## Sentence detection internals

`_get_sentence_boundaries()` does several things on top of raw Punkt:

### 1. Quote/apostrophe normalization

Smart quotes are normalized to ASCII before tokenizing so Punkt's punctuation
rules behave predictably:

```
“ ” -> "    ‘ ’ -> '
```

Boundary offsets are computed against this normalized string but applied back to
the original text, so the visible characters are never altered.

### 2. Per-language tokenizer cache

Punkt tokenizers are built once per language and cached in `_TOKENIZER_CACHE`.
The English model is the fallback when a language-specific model is missing.

### 3. Abbreviation set

The tokenizer's abbreviation set is assembled from three sources:

- **Inherited** abbreviations from Punkt's pretrained model (`Jan.`, `Corp.`,
  `Gen.`, US state abbreviations, etc.).
- A **curated `extra` list** of editorial/book abbreviations Punkt tends to
  over-split on (`Mr.`, `Mrs.`, `Dr.`, `St.`, `p.m.`, `U.S.`, ...).
- **Single letters `a`–`z`**, so initials like `Mr. Y.` are not treated as
  sentence ends.

### 4. Manual merge pass (`mark_sentence.py:691`)

Punkt sometimes *over-splits*. After tokenizing, a manual pass walks adjacent
boundary pairs and merges them back together in two cases:

- **Abbreviation**: the first span ends in `.`, and the word immediately before
  the period is in the abbreviation set. A backward scan extracts that word.
  A guard skips contractions like `won't.` (the backward scan stops at the
  apostrophe and would otherwise see a fake single-letter token `t`).
- **Possessive**: the boundary falls right before `'` followed by a lowercase
  letter (e.g. a name's possessive).

### 5. Trailing whitespace

Each boundary's end is extended to swallow trailing whitespace, so the space that
follows a sentence travels with that sentence rather than starting the next one.

## Known edge cases

The inline `test_your_case` assertions in `__main__` lock in the intended
behaviour. Representative cases:

| Input (abbreviated) | Why it's handled specially |
| --- | --- |
| `...than normal.” The merchant...` | Split after a closing smart quote that follows a sentence end. |
| `Mr. Needlemouse?` | `Mr.` is a curated abbreviation; do **not** split before the capitalized name. |
| `7:30 p.m. in Peter Main's...` | `p.m.` must not end the sentence. |
| `searching for it—looking cool,` | Em-dash creates a phrase boundary within the sentence. |
| `Mr. Y.'s speech,` | Single-letter initial `Y.` must not split; needs single-letter abbreviations. |
| `a meaningful moment in U.S. history.` | Dotted abbreviation `U.S.` stays intact. |
| `...heard anything of you … must come...` | Ellipsis acts as a sentence break. |
| `"Hello," she said.` | Split after the closing quote + comma. |
| `...but I won't. I don't feel...` | Contraction `won't.` is still a real sentence end (apostrophe guard). |
| `and Al was seriously ill. I could see...` | Real sentence break after `ill.` — see below. |

Run the assertions with:

```
python mark_sentence.py
```

Each case prints its detected boundaries and segments, then asserts the expected
split. A failure raises `AssertionError` naming the mismatch.

## The `ill` fix (debug writeup)

### Symptom

```python
text = 'and Al was seriously ill. I could see that in a single glance.'
# expected: ['and Al was seriously ill. ', 'I could see that in a single glance.']
# actual:   single segment spanning the whole string
```

### Root cause

Punkt's pretrained English model inherits `ill` as an abbreviation — it is the
traditional abbreviation for *Illinois*. When `ill.` is followed by the single
capital letter `I`, Punkt's built-in initial detection treats the sequence as
`<abbreviation>. <initial>` and **declines to split**. The sentence boundary
never appears, so the orchestrator emits one segment.

### Approaches rejected

- **Guard the merge pass** (only merge across a capitalized next word for
  "trusted" abbreviations): broke `Mr. Needlemouse`, which legitimately merges
  across a capitalized name.
- **Remove single-letter abbreviations** `a`–`z`: broke `Mr. Y.'s`, which relies
  on `Y.` being treated as an initial.

Both touched machinery that other passing cases depend on. The break was upstream
in tokenizer construction, not in the merge pass.

### Fix

Drop the one inherited abbreviation that is also an ordinary English word
(`mark_sentence.py:681`):

```python
params.abbrev_types.difference_update({"ill"})
```

### Tradeoff

`ill` is the standout false positive: a high-frequency common word that collides
with an inherited abbreviation. Other inherited abbreviations (`Jan.`, `Corp.`,
`Gen.`, state abbreviations, ...) are rarely written as plain lowercase words
mid-prose followed by a sentence break, so they are kept. If another common-word
collision surfaces later, add it to the same `difference_update({...})` set.

## How to extend

- **Add a regression case**: append a `text` / `expected_segments` pair and a
  `test_your_case(...)` call in the `__main__` block, then run
  `python mark_sentence.py`.
- **Add an abbreviation**: extend the `extra` list in `_get_sentence_boundaries()`.
- **Remove an inherited false-positive abbreviation**: add it to the
  `difference_update({...})` set near `mark_sentence.py:681`.

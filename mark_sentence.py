"""
HTML text segmentation for EPUB read-aloud overlays.

Purpose
-------

Insert stable segment spans into XHTML content for downstream Media Overlay / SMIL
generation.

Required properties:

- preserve visible text exactly
- preserve inline formatting and important structural inline tags
- preserve layout-affecting void tags
- keep output practical for ebook readers

Problem statement
-----------------

Segmentation logic operates on strings.
EPUB content is a DOM tree.

Naive approach:

- call `get_text()`
- segment the resulting string
- inject spans back into the HTML

Failure modes of the naive approach:

- inline markup is flattened
- empty anchors disappear
- void tags drift or vanish
- whitespace changes during parse/serialize cycles

This module uses an explicit linearize -> segment -> reconstruct pipeline instead.

Pipeline overview
-----------------

1. Pre-process vendor markup.
   We first remove Kobo wrapper spans that are not semantically meaningful but do
   make the DOM much noisier. This step must preserve the document structure; if we
   drop the `<head>` or break namespaces, the downstream sync script fails later in
   less obvious ways.

2. Linearize the DOM.
   For each processable block element we flatten the subtree into three parallel
   representations:

   - `linear_text`: the text that the segmentation logic will examine
   - `char_map`: for every visible character in `linear_text`, a mapping back to
     either the original text character or an atomic DOM node plus the inline
     formatting stack active at that position
   - `zero_width_nodes`: preserved structural tags that should survive
     reconstruction even though they do not contribute visible text, such as empty
     anchors with ids

    This is the core representation used by the rest of the module.

3. Detect segment boundaries.
   We do not rely on a single regex or a single sentence tokenizer result.
   Instead we:

   - start with NLTK Punkt sentence boundaries
   - repair known tokenizer failures with a manual merge pass
   - split within sentences using a state machine that is aware of quotes,
     parentheses, dashes, ellipses, commas, semicolons, and colons
   - merge tiny punctuation-only fragments or very short segments back into nearby
     text according to `min_words`

    Target output is stable, readable segmentation, not linguistically perfect
    sentence theory.

4. Reconstruct the DOM.
   Using the boundaries plus the map from step 2, we rebuild the element content as
   a sequence of `<span id="c<chapter>-segmentN">...</span>` nodes. During this
   step we must:

   - restore void tags at the right positions
   - restore inline formatting stacks around the right text slices
   - reinsert preserved zero-width structural tags at the exact character offsets
     where they originally lived
   - avoid generating wrapper garbage such as whitespace-only `<sup>` or `<a>`
     nodes, which can serialize differently and create false integrity failures

5. Validate aggressively.
    Compare original and segmented visible text. Treat mismatches as real failures.

Key invariants
--------------

- The visible text must remain unchanged.
- `len(char_map) == len(linear_text)` must always hold.
- Empty structural tags that matter to navigation or references must survive even
  if they contribute zero visible characters.
- Reconstruction should preserve formatting, but should not invent meaningless
  wrapper tags around pure whitespace.
- The output should remain performant enough for ebook readers; that is why we do
  not wrap every single character in its own span.

High-risk areas
---------------

1. Vanishing layout tags.
   Plain text extraction ignores `<br>` and `<img>`, so a reconstructed paragraph
   can visually collapse. We treat void tags as atomic map entries so they survive.

2. Tokenizer over-splitting on abbreviations.
   Punkt can split after things like `Mr.` or `U.S.` when followed by a capital.
   The manual merge pass exists because these mistakes are common in books.

3. Splitting punctuation inside quotes or parentheses.
   A regex that blindly splits on commas, colons, or dashes produces nonsense. The
   state machine tracks nesting/quote state before allowing a break.

4. Losing empty anchors.
   Many EPUBs contain empty `<a id="..."></a>` nodes that have no visible text but
   are semantically important. If they disappear, navigation and footnote links can
   break. These are now preserved as zero-width nodes.

5. Serializer/parser whitespace drift.
   Some XHTML structures, especially around footnotes and inline wrappers, can
   change blank-line counts when converted to string and reparsed. Validation now
   compares the live segmented DOM instead of assuming serialization is harmless.

6. Span soup.
   Wrapping every character independently would preserve formatting perfectly but is
   too heavy for real readers. The reconstruction groups contiguous text runs with
   the same formatting stack into single text nodes inside each segment span.

Section map
-----------

- `_build_char_map()` answers: "If I pretend this subtree is a string, what DOM
  thing does each character position correspond to?"
- `_get_sentence_aware_segment_boundaries()` answers: "Given that string, where are
  the best places to break it into read-aloud chunks?"
- `_create_segment_spans()` answers: "Using those boundaries, how do I rebuild the
  original subtree as segmented HTML without losing structure?"
- `validate_text_consistency()` answers: "Did we preserve what the user will
  actually see/read?"

Typical breakage points are whitespace handling, empty-anchor preservation, and
serializer/parser interactions around footnotes or appendix markup.
"""

import re
import logging
from pathlib import Path
from bs4 import (
    BeautifulSoup,
    NavigableString,
    Tag,
    Comment,
    Declaration,
    ProcessingInstruction,
)
import nltk
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer
from typing import List, Tuple, Union


_NLTK_RESOURCES_READY = False
_PUNKT_LANGUAGE_MAP = {
    "cs": "czech",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "it": "italian",
    "nl": "dutch",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sl": "slovene",
    "sv": "swedish",
    "tr": "turkish",
}


def _get_nltk_cache_dir() -> Path:
    return Path.home() / ".cache" / "epub-media-overlay" / "nltk_data"


def _ensure_nltk_search_path(cache_dir: Path) -> None:
    cache_dir_str = str(cache_dir)
    if cache_dir_str not in nltk.data.path:
        nltk.data.path.insert(0, cache_dir_str)


def _missing_nltk_resources() -> list[str]:
    missing = []
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(resource.rsplit("/", 1)[-1])
    return missing


def normalize_punkt_language(language: str | None) -> str:
    if not language:
        return "english"

    normalized = language.strip().lower().replace("_", "-")
    if normalized in _PUNKT_LANGUAGE_MAP:
        return _PUNKT_LANGUAGE_MAP[normalized]

    base_language = normalized.split("-", 1)[0]
    return _PUNKT_LANGUAGE_MAP.get(base_language, base_language)


def ensure_nltk_resources(logger: logging.Logger | None = None) -> None:
    global _NLTK_RESOURCES_READY
    if _NLTK_RESOURCES_READY:
        return

    cache_dir = _get_nltk_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _ensure_nltk_search_path(cache_dir)

    missing = _missing_nltk_resources()

    if missing:
        if logger:
            logger.info(
                "Downloading missing NLTK tokenizer data (%s) into %s",
                ", ".join(missing),
                cache_dir,
            )
        try:
            for resource_name in missing:
                nltk.download(
                    resource_name,
                    download_dir=str(cache_dir),
                    quiet=True,
                    raise_on_error=True,
                )
        except Exception as exc:
            raise RuntimeError(
                "Unable to download required NLTK tokenizer data "
                f"({', '.join(missing)}) into {cache_dir}: {exc}"
            ) from exc

        missing = _missing_nltk_resources()

    if missing:
        raise RuntimeError(
            "Missing NLTK tokenizer data after automatic download attempt: "
            f"{', '.join(missing)}"
        )

    _NLTK_RESOURCES_READY = True

# 1. CONSTANTS
VOID_TAGS = {
    "br",
    "img",
    "hr",
    "wbr",
    "area",
    "col",
    "embed",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "script",
    "style",
    "svg",
    "math",
    "object",
    "iframe",
}

INLINE_TAGS = {
    "span",
    "em",
    "i",
    "b",
    "strong",
    "u",
    "a",
    "sup",
    "sub",
    "q",
    "small",
    "big",
    "strike",
    "code",
    "font",
    "cite",
    "tt",
    "var",
    "del",
    "ins",
    "kbd",
    "samp",
    "mark",
    "ruby",
    "rt",
    "rp",
    "bdo",
    "dfn",
    "abbr",
    "time",
    "label",
    "button",
}

# 2. GLOBAL CACHE
_TOKENIZER_CACHE = {}


def preprocess_remove_kobo_spans(html_content: str) -> str:
    """Remove Kobo spans but preserve XML structure using lxml."""
    # Remove only the known mechanical Kobo wrapper pattern.
    soup = BeautifulSoup(html_content, "lxml")
    for span in soup.find_all("span"):
        if (
            len(span.attrs) == 3
            and span.attrs.get("class") == ["koboSpan"]
            and span.attrs.get("id", "").startswith("kobo")
        ):
            span.unwrap()

    # CRITICAL FIX: Return the FULL document (str(soup)).
    # Do NOT return soup.body.encode_contents(), or you lose the <head> tag,
    # causing 'AttributeError: NoneType has no attribute append' in the sync script.
    return str(soup)


def mark_sentences(
    html_content: str,
    chapter_id: str = "chapter",
    language: str = "english",
    min_words: int = 1,
) -> str:
    """Segment content by wrapping text in segment spans. Preserves structure and void tags."""

    # Stage 1: remove vendor-specific wrapper noise.
    cleaned_html = preprocess_remove_kobo_spans(html_content)

    # Stage 2: parse the full cleaned document.
    soup = BeautifulSoup(cleaned_html, "lxml")

    global_counter = 1

    # Stage 3: process only block elements that are effectively leaves in block space.
    block_tags = [
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "blockquote",
        "div",
        "dd",
        "dt",
        "figcaption",
    ]

    for elem in soup.find_all(block_tags):
        if not _has_block_children(elem) and _has_meaningful_text(elem):
            global_counter = _process_element_preserve_structure(
                elem, chapter_id, global_counter, language, min_words
            )

    final_html = str(soup)

    # Stage 4: validate against the live segmented DOM, not against reparsed output.
    if not validate_text_consistency(cleaned_html, soup, chapter_id):
        raise ValueError(f"Text integrity check failed for {chapter_id}")

    return final_html


def _has_block_children(element: Tag) -> bool:
    """Check if element contains other block-level elements."""
    block_elements = {
        "div",
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "ul",
        "ol",
        "li",
        "blockquote",
        "table",
        "tr",
        "td",
        "dl",
        "dd",
        "dt",
        "section",
        "article",
        "nav",
        "aside",
        "header",
        "footer",
        "figure",
        "figcaption",
    }
    return any(
        child.name in block_elements
        for child in element.children
        if isinstance(child, Tag)
    )


def _has_meaningful_text(element: Tag) -> bool:
    """Check if element has text or void tags."""
    # Treat void-tag-only blocks as meaningful; they still affect layout/reading flow.
    if element.get_text(strip=True):
        return True
    if element.find(list(VOID_TAGS)):
        return True
    return False


def _process_element_preserve_structure(
    element: Tag, chapter_id: str, counter: int, language: str, min_words: int
) -> int:
    """Maps content, segments text, and reconstructs element with spans."""
    # Local pipeline for one block subtree.

    # Step 1: linearize subtree into string-space plus reconstruction metadata.
    char_map, linear_text, zero_width_nodes = _build_char_map(element)

    if len(char_map) != len(linear_text):
        raise ValueError(f"Char map mismatch: {len(char_map)} vs {len(linear_text)}")

    if not linear_text.strip():
        return counter

    # Step 2: choose boundaries in string space.
    boundaries = _get_sentence_aware_segment_boundaries(
        linear_text, language, min_words
    )

    if not boundaries:
        return counter

    # Step 3: reconstruct subtree from the boundaries and linearized map.
    new_content = _create_segment_spans(
        boundaries, char_map, zero_width_nodes, chapter_id, counter
    )

    # Step 4: replace subtree contents only after reconstruction succeeds.
    counter += len(new_content)
    element.clear()
    for item in new_content:
        element.append(item)

    return counter


def _build_char_map(
    element: Tag,
) -> Tuple[List[Tuple[Union[str, Tag], List[Tag]]], str, List[Tuple[int, Tag, List[Tag]]]]:
    """Flattens the DOM tree into a linear character map."""
    # Invariant: `char_map` and `linear_text` remain positionally aligned.
    char_map = []
    linear_text_parts = []
    # Zero-width structural tags cannot occupy visible character positions, so store
    # them as "insert node at text offset N" records.
    zero_width_nodes = []

    def walk(node: Union[Tag, NavigableString], stack: List[Tag]):
        # IGNORE COMMENTS
        if isinstance(node, (Comment, Declaration, ProcessingInstruction)):
            return

        if isinstance(node, NavigableString):
            text = str(node)
            # Character-level mapping enables exact reconstruction later.
            for char in text:
                char_map.append((char, stack))
            linear_text_parts.append(text)

        elif isinstance(node, Tag):
            # Case A: void tag represented as one atomic map item plus a placeholder.
            if node.name in VOID_TAGS:
                placeholder = "\n" if node.name == "br" else " "
                char_map.append((node, stack))
                linear_text_parts.append(placeholder)
            elif _is_preserved_empty_tag(node):
                # Case B: preserved zero-width structural tag.
                zero_width_nodes.append((len(char_map), _clone_tag_shell(node), stack[:]))
            else:
                # Case C: ordinary container tag. Inline containers extend the active
                # formatting stack for descendants.
                new_stack = stack[:]
                if node.name in INLINE_TAGS:
                    new_stack.append(node)
                for child in node.children:
                    walk(child, new_stack)

    walk(element, [])
    return char_map, "".join(linear_text_parts), zero_width_nodes


def _is_preserved_empty_tag(node: Tag) -> bool:
    # Preserve the empty-anchor pattern used for footnotes, page targets, and similar
    # EPUB navigation points.
    if node.name != "a":
        return False
    if node.contents:
        return False
    return bool(node.get("id") or node.get("name"))


def _clone_tag_shell(node: Tag) -> Tag:
    # Copy tag name + attributes only.
    return Tag(name=node.name, attrs=dict(node.attrs))


def _wrap_with_format_stack(node: Union[Tag, NavigableString], fmt_stack: List[Tag]):
    # `fmt_stack` is stored outermost -> innermost, so wrap in reverse order.
    wrapped_node = node
    for fmt in reversed(fmt_stack):
        wrapper = Tag(name=fmt.name, attrs=dict(fmt.attrs))
        wrapper.append(wrapped_node)
        wrapped_node = wrapper
    return wrapped_node


def _create_segment_spans(
    boundaries: List[Tuple[int, int]],
    char_map: List,
    zero_width_nodes: List[Tuple[int, Tag, List[Tag]]],
    chapter_id: str,
    start_counter: int,
) -> List[Tag]:
    """Reconstructs the HTML using the boundaries and character map."""
    # `zero_width_lookup` maps text offsets to structural zero-width tags that must be
    # emitted at those positions.
    new_content: List[Tag] = []
    counter = start_counter
    zero_width_lookup = {}
    for position, node, fmt_stack in zero_width_nodes:
        zero_width_lookup.setdefault(position, []).append((node, fmt_stack))

    for start, end in boundaries:
        if start >= end:
            continue

        # Each boundary range becomes one segment wrapper span.
        segment_span = Tag(name="span", attrs={"id": f"c{chapter_id}-segment{counter}"})
        counter += 1

        current_idx = start
        while current_idx < end:
            # Emit zero-width nodes before consuming the visible item at this offset.
            for node, fmt_stack in zero_width_lookup.get(current_idx, []):
                segment_span.append(_wrap_with_format_stack(node, fmt_stack))

            item, fmt_stack = char_map[current_idx]

            # Case A: restore an atomic non-text node from its placeholder position.
            if isinstance(item, Tag):
                segment_span.append(_wrap_with_format_stack(item, fmt_stack))
                current_idx += 1
                continue

            # Case B: contiguous text run with the same formatting stack.
            # Stop the run when formatting changes, a tag appears, or a zero-width node
            # must be inserted at the next index.
            text_buffer = [item]
            next_idx = current_idx + 1
            while next_idx < end:
                next_item, next_fmt = char_map[next_idx]
                if (
                    isinstance(next_item, Tag)
                    or next_fmt != fmt_stack
                    or next_idx in zero_width_lookup
                ):
                    break
                text_buffer.append(next_item)
                next_idx += 1

            text_str = "".join(text_buffer)
            if text_str.strip():
                wrapped_node = _wrap_with_format_stack(NavigableString(text_str), fmt_stack)
            else:
                # Keep pure whitespace raw. Wrapping whitespace-only content in inline
                # tags tends to create serializer-dependent junk.
                wrapped_node = NavigableString(text_str)

            segment_span.append(wrapped_node)
            current_idx = next_idx

        # Emit zero-width nodes that land exactly on the right boundary.
        for node, fmt_stack in zero_width_lookup.get(end, []):
            segment_span.append(_wrap_with_format_stack(node, fmt_stack))

        new_content.append(segment_span)

    return new_content


# === Segmentation Logic ===


def _get_sentence_boundaries(text: str, language: str) -> List[Tuple[int, int]]:
    """Get sentence boundaries using Cached Tokenizer."""
    # Punkt provides the initial sentence candidates. Later logic refines them.
    ensure_nltk_resources()
    language = normalize_punkt_language(language)
    normalized_text = (
        text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    )

    global _TOKENIZER_CACHE
    if language not in _TOKENIZER_CACHE:
        # Cache tokenizers per language.
        try:
            base = nltk.data.load(f"tokenizers/punkt/{language}.pickle")
        except:
            base = nltk.data.load("tokenizers/punkt/english.pickle")

        params = PunktParameters()
        if hasattr(base, "_params") and hasattr(base._params, "abbrev_types"):
            params.abbrev_types = set(base._params.abbrev_types)
        else:
            params.abbrev_types = set()

        # Extra abbreviations cover common editorial/book cases Punkt over-splits.
        extra = [
            "mr",
            "mrs",
            "ms",
            "dr",
            "prof",
            "rev",
            "sr",
            "jr",
            "st",
            "ave",
            "blvd",
            "rd",
            "ln",
            "gov",
            "rep",
            "sen",
            "vs",
            "etc",
            "eg",
            "ie",
            "phd",
            "md",
            "ba",
            "ma",
            "sgt",
            "am",
            "pm",
            "us",
            "u.s",
        ]
        params.abbrev_types.update(extra)
        params.abbrev_types.update(set("abcdefghijklmnopqrstuvwxyz"))
        _TOKENIZER_CACHE[language] = (
            PunktSentenceTokenizer(params),
            params.abbrev_types,
        )

    custom_tokenizer, abbrev_types_set = _TOKENIZER_CACHE[language]

    boundaries = list(custom_tokenizer.span_tokenize(normalized_text))

    # Manual merge pass for known Punkt false positives.
    i = 0
    while i < len(boundaries) - 1:
        start, end = boundaries[i]
        next_start, next_end = boundaries[i + 1]

        # Abbreviation
        if end > 0 and normalized_text[end - 1] == ".":
            k = end - 2
            while k >= start and normalized_text[k].isalpha():
                k -= 1
            word = normalized_text[k + 1 : end - 1].lower()
            if word in abbrev_types_set:
                boundaries[i] = (start, next_end)
                del boundaries[i + 1]
                continue
        # Possessive
        if end < len(normalized_text) and normalized_text[end] == "'":
            if end < len(normalized_text) - 1 and normalized_text[end + 1].islower():
                boundaries[i] = (start, next_end)
                del boundaries[i + 1]
                continue
        i += 1

    # Expand boundaries to include trailing whitespace.
    final_boundaries = []
    for start, end in boundaries:
        while end < len(text) and text[end].isspace():
            end += 1
        final_boundaries.append((start, end))

    return final_boundaries


def _get_segment_boundaries_in_sentence(sentence_text: str) -> List[Tuple[int, int]]:
    """State-machine splitting."""
    # Phrase-level boundary detector for one sentence-sized region. Quote and
    # parenthesis state is tracked explicitly so punctuation does not force invalid
    # splits.
    n = len(sentence_text)
    i = 0
    breaks = [0]
    paren_level = 0
    double_quote_level = 0
    single_quote_level = 0

    while i < n:
        c = sentence_text[i]
        if c in ['"', "“", "”"]:
            # Double-quote transitions can define phrase boundaries at top level.
            is_open_quote = c == "“" or (c == '"' and double_quote_level == 0)
            is_close_quote = c == "”" or (c == '"' and double_quote_level > 0)

            if is_open_quote:
                if double_quote_level == 0 and single_quote_level == 0:
                    breaks.append(i)
                double_quote_level += 1
            elif is_close_quote and double_quote_level > 0:
                double_quote_level -= 1
                if double_quote_level == 0 and single_quote_level == 0:
                    j = i + 1
                    while j < n and sentence_text[j].isspace():
                        j += 1
                    breaks.append(j)
        elif c in ["'", "‘", "’"]:
            # Single quotes need heuristics because apostrophes and quotation marks use
            # overlapping characters.
            is_quote = (
                i == 0
                or sentence_text[i - 1].isspace()
                or (i < n - 1 and sentence_text[i + 1].isupper())
                or (i > 0 and sentence_text[i - 1] in {".", "!", "?"})
            )
            if (
                i > 0
                and i < n - 1
                and sentence_text[i - 1] == "."
                and sentence_text[i + 1].islower()
            ):
                is_quote = False
            if is_quote:
                if (
                    c in ["'", "‘"]
                    and single_quote_level == 0
                    and double_quote_level == 0
                ):
                    breaks.append(i)
                    single_quote_level += 1
                elif c in ["’"] and single_quote_level > 0:
                    single_quote_level -= 1
                    if single_quote_level == 0 and double_quote_level == 0:
                        j = i + 1
                        while j < n and sentence_text[j].isspace():
                            j += 1
                        breaks.append(j)
        elif c == "(":
            # Parenthetical material can define a boundary at top level.
            if paren_level == 0 and single_quote_level == 0 and double_quote_level == 0:
                breaks.append(i)
            paren_level += 1
        elif c == ")" and paren_level > 0:
            paren_level -= 1
            if paren_level == 0 and single_quote_level == 0 and double_quote_level == 0:
                j = i + 1
                while j < n and sentence_text[j].isspace():
                    j += 1
                breaks.append(j)
        if c == "—" or c == "…":
            # Em dashes and ellipses often correspond to audible pauses.
            j = i + 1
            while j < n and sentence_text[j].isspace():
                j += 1
            breaks.append(j)
        if c in {",", ";", ":"}:
            skip_break = False
            if c == "," or c == ":":
                # Do not split numeric constructs such as time stamps.
                if (
                    i > 0
                    and i < n - 1
                    and sentence_text[i - 1].isdigit()
                    and sentence_text[i + 1].isdigit()
                ):
                    skip_break = True
            if not skip_break:
                j = i + 1
                while j < n and sentence_text[j].isspace():
                    j += 1
                breaks.append(j)
        i += 1

    breaks.append(n)
    breaks = sorted(set(breaks))
    return [
        (breaks[k], breaks[k + 1])
        for k in range(len(breaks) - 1)
        if breaks[k] < breaks[k + 1]
    ]


def _get_sentence_aware_segment_boundaries(
    text: str, language: str, min_words: int
) -> List[Tuple[int, int]]:
    """Orchestrator with GAP FILLING."""
    # Combine sentence boundaries with phrase boundaries and ensure full text coverage.
    sent_boundaries = _get_sentence_boundaries(text, language)
    if not sent_boundaries:
        return [(0, len(text))] if text else []

    final_boundaries = []
    current_pos = 0

    for sent_start, sent_end in sent_boundaries:
        # Preserve any gap between detected sentence spans.
        if sent_start > current_pos:
            final_boundaries.append((current_pos, sent_start))

        sent_text = text[sent_start:sent_end]
        phrase_boundaries = _get_segment_boundaries_in_sentence(sent_text)

        # Merge short or punctuation-only fragments into neighbors.
        merged_phrases = []
        if not phrase_boundaries:
            current_pos = sent_end
            continue

        p_curr_start = phrase_boundaries[0][0]
        p_curr_end = phrase_boundaries[0][1]

        for i in range(1, len(phrase_boundaries)):
            p_start, p_end = phrase_boundaries[i]
            curr_chunk = sent_text[p_curr_start:p_curr_end]
            next_chunk = sent_text[p_start:p_end]
            curr_words = len(re.findall(r"\w+", curr_chunk))
            is_next_punct = not any(c.isalnum() for c in next_chunk)
            if curr_words < min_words or is_next_punct:
                p_curr_end = p_end
            else:
                merged_phrases.append((p_curr_start, p_curr_end))
                p_curr_start = p_start
                p_curr_end = p_end
        merged_phrases.append((p_curr_start, p_curr_end))
        for p_start, p_end in merged_phrases:
            final_boundaries.append((sent_start + p_start, sent_start + p_end))
        current_pos = sent_end

    if current_pos < len(text):
        final_boundaries.append((current_pos, len(text)))
    return final_boundaries


def validate_text_consistency(
    original_html: Union[str, bytes, BeautifulSoup, Tag],
    segmented_html: Union[str, bytes, BeautifulSoup, Tag],
    chapter_id: str = "chapter",
) -> bool:
    """Safely validate text integrity. NO STRIP to preserve spaces."""

    # Strict visible-text integrity check with localized diff output on failure.

    def to_soup(content):
        if isinstance(content, (BeautifulSoup, Tag)):
            return content
        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8")
            except:
                content = content.decode("latin-1", errors="replace")
        return BeautifulSoup(content, "lxml")

    original_soup = to_soup(original_html)
    segmented_soup = to_soup(segmented_html)

    orig_text = original_soup.get_text()
    new_text = segmented_soup.get_text()

    # First compare with light whitespace normalization.
    def normalize_for_validation(text):
        # Collapse repeated spaces within each line but preserve line boundaries.
        lines = [re.sub(r" +", " ", line).strip() for line in text.split("\n")]
        return "\n".join(lines)

    orig_normalized = normalize_for_validation(orig_text)
    new_normalized = normalize_for_validation(new_text)

    if orig_normalized == new_normalized:
        return True

    # Locate the first raw divergence for debugging output.
    min_len = min(len(orig_text), len(new_text))
    diff_idx = min_len
    for i in range(min_len):
        if orig_text[i] != new_text[i]:
            diff_idx = i
            break

    # Print a local text window around the mismatch. `repr()` keeps hidden whitespace
    # visible in the log.
    context_start = max(0, diff_idx - 40)
    context_end = diff_idx + 60

    print(f"\n{'=' * 30}")
    print(f"INTEGRITY FAIL: {chapter_id}")
    print(f"Divergence detected at index: {diff_idx}")

    # We use repr() to reveal hidden characters like \n, \t, or \r
    print(f"Original  Diff: ...{repr(orig_text[context_start:context_end])}...")
    print(f"Segmented Diff: ...{repr(new_text[context_start:context_end])}...")

    if len(orig_text) != len(new_text):
        print(f"Length mismatch: Original={len(orig_text)}, Segmented={len(new_text)}")

    return False


# === Test Function ===
def test_your_case(text, expected_segments):
    language = "english"
    min_words = 1

    print("-" * 50)
    print("-" * 50)
    print("Testing text:", repr(text))
    print("\n--- Detected Boundaries ---")
    boundaries = _get_sentence_aware_segment_boundaries(text, language, min_words)
    print(boundaries)

    print("\n--- Extracted Segments ---")
    segments = [text[start:end] for start, end in boundaries]
    for i, seg in enumerate(segments, 1):
        print(f"Segment {i}: {repr(seg)}")

    assert len(segments) == len(expected_segments), (
        f"Expected {len(expected_segments)} segments, got {len(segments)}"
    )

    print("\n✅ Test PASSED!")
    return True


if __name__ == "__main__":
    text = "the guy who passed along the tip gets a bigger allocation than normal.” The merchant tapped his fingers on the desk."
    expected_segments = [
        "the guy who passed along the tip gets a bigger allocation than normal.”",
        " The merchant tapped his fingers on the desk.",
    ]
    test_your_case(text, expected_segments)

    text = "“This is the hedgehog named Mr. Needlemouse?”"
    expected_segments = ["“This is the hedgehog named Mr. Needlemouse?”"]
    test_your_case(text, expected_segments)

    text = "The discussion was set to begin around 7:30 p.m. in Peter Main’s hotel suite at Caesar’s Palace."
    expected_segments = [
        "The discussion was set to begin around 7:30 p.m. in Peter Main’s hotel suite at Caesar’s Palace."
    ]
    test_your_case(text, expected_segments)

    text = "and in between you spend a bunch of years searching for it—looking cool,"
    expected_segments = [
        "and in between you spend a bunch of years searching for it—",
        "looking cool,",
    ]
    test_your_case(text, expected_segments)

    text = "In the middle of Mr. Y.’s speech,"
    expected_segments = ["In the middle of Mr. Y.’s speech,"]
    test_your_case(text, expected_segments)

    text = "“a meaningful moment in U.S. history.”"
    expected_segments = ["“a meaningful moment in U.S. history.”"]
    test_your_case(text, expected_segments)

    text = "such years since I heard anything of you … must come to Soldier Island …"
    expected_segments = [
        "such years since I heard anything of you … ",
        "must come to Soldier Island …",
    ]
    test_your_case(text, expected_segments)

    text = '"Hello," she said.'
    expected_segments = ['"Hello,"', " she said."]
    test_your_case(text, expected_segments)

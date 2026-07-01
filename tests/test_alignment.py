"""Unit tests for the token-level audio<->HTML alignment step in pipeline_core.

These exercise the pure functions that turn ASR word timings + HTML segment
tokens into per-segment time envelopes:

  - split_into_match_tokens   (hyphen/dash aware tokenization)
  - build_raw_matches         (difflib equal-runs -> per-segment envelope)
  - build_segment_match_list  (segment-ordered match list; drops zero-match segs)
  - fill_interior_segment_gaps(interpolate interior segments the aligner missed)
  - finalize_segment_timestamps (gap closing + boundary anchoring)

Regression anchor: the real "dry-swallowed" bug from chapter 9 (010.json), where
the hyphenated HTML word ("dry-swallowed") did not match the two ASR words
("dry", "swallowed"), so its segment was dropped from the SMIL.

Run:
  /Users/f2pgod/Documents/spyder312/bin/python -m pytest tests/test_alignment.py -q
or:
  /Users/f2pgod/Documents/spyder312/bin/python tests/test_alignment.py
"""

# Bootstrap: allow running this file directly.
# Source modules are imported by bare name below, so the repo root must be on
# sys.path before those imports. Under pytest, pyproject's pythonpath handles this.
import sys as _sys
from pathlib import Path as _Path

_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import difflib
import os
import tempfile
import zipfile

from bs4 import BeautifulSoup

import pipeline_core as pc


# --- helpers ---------------------------------------------------------------


def make_segments(seg_ids):
    """Build BeautifulSoup span tags with ids, in document order.

    Mirrors what `soup.select('[id*="-segment"]')` returns: objects exposing
    `.get("id")` and iterated in order.
    """
    soup = BeautifulSoup("<root/>", "lxml")
    tags = []
    for seg_id in seg_ids:
        tag = soup.new_tag("span", id=seg_id)
        tags.append(tag)
    return tags


def html_tokens_from(spec):
    """spec: list of (seg_id, "raw segment text"). Returns flattened html_tokens
    using the same tokenization as load_html_segments_and_tokens."""
    html_tokens = []
    for seg_index, (seg_id, text) in enumerate(spec):
        for word in text.split():
            for token in pc.split_into_match_tokens(word):
                html_tokens.append(
                    {"token": token, "seg_id": seg_id, "seg_index": seg_index}
                )
    return html_tokens


def audio_tokens_from(words):
    """words: list of (word, start, end). Returns flattened audio_tokens using
    the same tokenization as load_audio_tokens (hyphen-splitting, dup start/end)."""
    audio_tokens = []
    for word, start, end in words:
        for token in pc.split_into_match_tokens(word):
            audio_tokens.append(
                {"token": token, "word": word, "start": start, "end": end}
            )
    return audio_tokens


def align(html_tokens, audio_tokens, segments):
    """Run the full fine-alignment helper chain and return the matched list
    (post interior-gap fill, pre finalize)."""
    matcher = difflib.SequenceMatcher(
        None,
        [t["token"] for t in html_tokens],
        [t["token"] for t in audio_tokens],
        autojunk=False,
    )
    raw_matches, *_ = pc.build_raw_matches(
        matcher.get_opcodes(), html_tokens, audio_tokens
    )
    matched = pc.build_segment_match_list(raw_matches, segments)
    return pc.fill_interior_segment_gaps(matched, segments)


def by_id(matched_list):
    return {m["id"]: m for m in matched_list}


# --- 1. tokenizer: split_into_match_tokens --------------------------------


def test_split_hyphenated_compound():
    assert pc.split_into_match_tokens("dry-swallowed") == ["dry", "swallowed"]


def test_split_preserves_plain_word():
    assert pc.split_into_match_tokens("walked") == ["walked"]


def test_split_does_not_break_apostrophes():
    # ASR emits "don't" as one word -> "dont"; HTML must produce the same single
    # token so they still match. Apostrophes must NOT cause a split.
    assert pc.split_into_match_tokens("don't") == ["dont"]
    assert pc.split_into_match_tokens("Harry's") == ["harrys"]


def test_split_strips_surrounding_punctuation():
    assert pc.split_into_match_tokens("swallowed.") == ["swallowed"]
    assert pc.split_into_match_tokens('"dry-swallowed,"') == ["dry", "swallowed"]


def test_split_abbreviation_collapses():
    # "U.S." has no hyphen and the dots are stripped -> single token "us".
    assert pc.split_into_match_tokens("U.S.") == ["us"]


def test_split_unicode_dashes():
    # en-dash, em-dash, non-breaking hyphen all split.
    assert pc.split_into_match_tokens("hot\u2013and\u2014dry") == ["hot", "and", "dry"]
    assert pc.split_into_match_tokens("dry\u2011mouthed") == ["dry", "mouthed"]


def test_split_multi_hyphen_compound():
    assert pc.split_into_match_tokens("re-election") == ["re", "election"]
    assert pc.split_into_match_tokens("tee-shirt") == ["tee", "shirt"]


def test_split_empty_and_pure_punctuation():
    assert pc.split_into_match_tokens("") == []
    assert pc.split_into_match_tokens("---") == []
    assert pc.split_into_match_tokens("--") == []


def test_split_leading_trailing_dash():
    # Leading/trailing dashes must not create empty tokens.
    assert pc.split_into_match_tokens("-dry-") == ["dry"]


# --- 2. end-to-end: the real "dry-swallowed" regression --------------------


def test_dry_swallowed_segment_is_matched():
    """The exact chapter-9 case. Word timings taken from 010.json.

    Before the fix the middle segment ("dry-swallowed.") matched nothing and was
    dropped. Now it must be present with an envelope spanning dry+swallowed.
    """
    spec = [
        ("seg116", "stuck them in my mouth,"),
        ("seg117", "dry-swallowed."),
        ("seg118", "Then I got up and walked slowly over to the Wall of Celebrity."),
    ]
    words = [
        ("stuck", 274.834, 275.094),
        ("them", 275.134, 275.234),
        ("in", 275.274, 275.354),
        ("my", 275.394, 275.514),
        ("mouth", 275.554, 275.815),
        ("dry", 275.975, 276.215),
        ("swallowed", 276.275, 276.735),
        ("then", 277.496, 277.636),
        ("i", 277.656, 277.696),
        ("got", 277.736, 277.876),
        ("up", 277.916, 277.996),
        ("and", 278.036, 278.116),
        ("walked", 278.156, 278.376),
        ("slowly", 278.436, 278.797),
    ]
    segments = make_segments(["seg116", "seg117", "seg118"])
    matched = by_id(align(html_tokens_from(spec), audio_tokens_from(words), segments))

    assert "seg117" in matched, "dry-swallowed segment must not be dropped"
    assert matched["seg117"]["start"] == 275.975  # start of 'dry'
    assert matched["seg117"]["end"] == 276.735  # end of 'swallowed'
    # And it must be a real matched segment, not an interpolated placeholder.
    assert not matched["seg117"].get("interpolated", False)


def test_dry_swallowed_neighbor_no_longer_absorbs_audio():
    """seg116 must end at 'mouth' (275.815), not swallow dry+swallowed."""
    spec = [
        ("seg116", "stuck them in my mouth,"),
        ("seg117", "dry-swallowed."),
        ("seg118", "Then I got up."),
    ]
    words = [
        ("stuck", 274.834, 275.094),
        ("them", 275.134, 275.234),
        ("in", 275.274, 275.354),
        ("my", 275.394, 275.514),
        ("mouth", 275.554, 275.815),
        ("dry", 275.975, 276.215),
        ("swallowed", 276.275, 276.735),
        ("then", 277.496, 277.636),
        ("i", 277.656, 277.696),
        ("got", 277.736, 277.876),
        ("up", 277.916, 277.996),
    ]
    segments = make_segments(["seg116", "seg117", "seg118"])
    matched = by_id(align(html_tokens_from(spec), audio_tokens_from(words), segments))
    assert matched["seg116"]["end"] == 275.815  # raw envelope ends at 'mouth'


def test_dry_swallowed_after_finalize_is_continuous():
    """After finalize, seg117 fills the gap up to seg118's start (no silent
    absorption by seg116) and ordering is preserved."""
    spec = [
        ("seg116", "stuck them in my mouth,"),
        ("seg117", "dry-swallowed."),
        ("seg118", "Then I got up."),
    ]
    words = [
        ("them", 275.134, 275.234),
        ("in", 275.274, 275.354),
        ("my", 275.394, 275.514),
        ("mouth", 275.554, 275.815),
        ("dry", 275.975, 276.215),
        ("swallowed", 276.275, 276.735),
        ("then", 277.496, 277.636),
        ("got", 277.736, 277.876),
        ("up", 277.916, 277.996),
    ]
    segments = make_segments(["seg116", "seg117", "seg118"])
    matched = align(html_tokens_from(spec), audio_tokens_from(words), segments)
    final = pc.finalize_segment_timestamps(matched, total_duration=999.0)
    final_by_id = {m["id"]: m for m in final}

    # gap-closing: seg116.end == seg117.start, seg117.end == seg118.start
    assert final_by_id["seg116"]["end"] == final_by_id["seg117"]["start"]
    assert final_by_id["seg117"]["end"] == final_by_id["seg118"]["start"]
    # seg117 spans real audio, end pushed to seg118 start (277.496)
    assert final_by_id["seg117"]["start"] == 275.975
    assert final_by_id["seg117"]["end"] == 277.496


# --- 3. safety net: interior gap interpolation -----------------------------


def test_interior_unmatched_segment_is_interpolated():
    """A segment whose tokens cannot match any audio word (here a number that ASR
    rendered as words) must still appear, interpolated between its neighbors."""
    spec = [
        ("segA", "He waited"),
        ("segB", "1958"),  # ASR said "nineteen fifty eight" -> no token match
        ("segC", "years later"),
    ]
    words = [
        ("He", 1.0, 1.2),
        ("waited", 1.3, 1.6),
        ("nineteen", 2.0, 2.3),
        ("fifty", 2.4, 2.6),
        ("eight", 2.7, 2.9),
        ("years", 3.0, 3.3),
        ("later", 3.4, 3.7),
    ]
    segments = make_segments(["segA", "segB", "segC"])
    matched = by_id(align(html_tokens_from(spec), audio_tokens_from(words), segments))

    assert "segB" in matched, "interior unmatched segment must be interpolated"
    assert matched["segB"].get("interpolated") is True
    # Interpolated point sits within the gap between segA.end and segC.start.
    assert matched["segA"]["end"] <= matched["segB"]["start"] <= matched["segC"]["start"]


def test_multiple_consecutive_interior_gaps_distributed():
    """Two consecutive unmatched interior segments split the gap evenly."""
    spec = [
        ("segA", "alpha"),
        ("segB", "1958"),  # unmatched
        ("segC", "12"),  # unmatched
        ("segD", "omega"),
    ]
    words = [
        ("alpha", 10.0, 10.5),
        ("nineteen", 11.0, 11.2),
        ("fifty", 11.3, 11.5),
        ("eight", 11.6, 11.8),
        ("twelve", 12.0, 12.2),
        ("omega", 14.0, 14.5),
    ]
    segments = make_segments(["segA", "segB", "segC", "segD"])
    matched = by_id(align(html_tokens_from(spec), audio_tokens_from(words), segments))

    assert "segB" in matched and "segC" in matched
    # gap is [segA.end=10.5, segD.start=14.0]; evenly into 3 -> ~11.667, ~12.833
    assert matched["segA"]["end"] < matched["segB"]["start"] < matched["segC"]["start"]
    assert matched["segC"]["start"] < matched["segD"]["start"]


def test_leading_unmatched_segment_not_fabricated():
    """A gap before the first matched segment has no anchor -> left dropped."""
    spec = [
        ("segA", "1958"),  # unmatched, leading
        ("segB", "years later"),
    ]
    words = [
        ("nineteen", 2.0, 2.3),
        ("fifty", 2.4, 2.6),
        ("eight", 2.7, 2.9),
        ("years", 3.0, 3.3),
        ("later", 3.4, 3.7),
    ]
    segments = make_segments(["segA", "segB"])
    matched = by_id(align(html_tokens_from(spec), audio_tokens_from(words), segments))
    assert "segA" not in matched  # not fabricated
    assert "segB" in matched


def test_trailing_unmatched_segment_not_fabricated():
    """A gap after the last matched segment has no forward anchor -> dropped."""
    spec = [
        ("segA", "years later"),
        ("segB", "1958"),  # unmatched, trailing
    ]
    words = [
        ("years", 3.0, 3.3),
        ("later", 3.4, 3.7),
        ("nineteen", 4.0, 4.3),
        ("fifty", 4.4, 4.6),
        ("eight", 4.7, 4.9),
    ]
    segments = make_segments(["segA", "segB"])
    matched = by_id(align(html_tokens_from(spec), audio_tokens_from(words), segments))
    assert "segA" in matched
    assert "segB" not in matched  # not fabricated


def test_fill_noop_when_all_matched():
    spec = [("segA", "alpha beta"), ("segB", "gamma delta")]
    words = [
        ("alpha", 1.0, 1.2),
        ("beta", 1.3, 1.5),
        ("gamma", 2.0, 2.2),
        ("delta", 2.3, 2.5),
    ]
    segments = make_segments(["segA", "segB"])
    matcher = difflib.SequenceMatcher(
        None,
        [t["token"] for t in html_tokens_from(spec)],
        [t["token"] for t in audio_tokens_from(words)],
        autojunk=False,
    )
    raw, *_ = pc.build_raw_matches(
        matcher.get_opcodes(), html_tokens_from(spec), audio_tokens_from(words)
    )
    matched = pc.build_segment_match_list(raw, segments)
    filled = pc.fill_interior_segment_gaps(matched, segments)
    assert len(filled) == len(matched) == 2
    assert all(not m.get("interpolated", False) for m in filled)


def assert_no_overlapping_clips(final_list):
    """Mirror the pipeline's overlapping_audio_clips validator: in document order,
    each emitted (positive-length) clip must not start before the previous ends."""
    emitted = sorted(
        (m for m in final_list if m["end"] > m["start"]),
        key=lambda m: m["segment_index"],
    )
    for prev, cur in zip(emitted, emitted[1:]):
        assert cur["start"] >= prev["end"], (
            f"overlap: {prev['id']} [{prev['start']},{prev['end']}] vs "
            f"{cur['id']} [{cur['start']},{cur['end']}]"
        )


def test_ellipsis_spans_with_negative_gap_no_overlap():
    """Reproduces the real ch026 segment559/562 failure: a span trailing into an
    ellipsis, two punctuation-only spans, then a continuation span whose audio
    starts BEFORE the first span's raw end (non-monotonic envelope).

    The punctuation spans must not be interpolated into the negative gap, and the
    surrounding real clips must not overlap after finalize.
    """
    spec = [
        ("seg559", "Is it"),
        ("seg560", "."),  # ellipsis -> zero tokens
        ("seg561", "."),  # ellipsis -> zero tokens
        ("seg562", "I couldn't get the name"),
    ]
    # Audio envelopes mirror the production values: 562 starts (1460.623) before
    # 559's raw end (1462.144).
    words = [
        ("is", 1460.463, 1461.0),
        ("it", 1461.2, 1462.144),
        ("i", 1460.623, 1460.9),  # 562 region begins earlier than 559 ends
        ("couldnt", 1461.5, 1461.9),
        ("get", 1462.0, 1462.3),
        ("the", 1462.4, 1462.6),
        ("name", 1462.7, 1463.406),
    ]
    segments = make_segments(["seg559", "seg560", "seg561", "seg562"])
    matched = align(html_tokens_from(spec), audio_tokens_from(words), segments)
    matched_ids = {m["id"] for m in matched}

    # No room in the negative gap -> ellipsis spans are not interpolated.
    assert "seg560" not in matched_ids
    assert "seg561" not in matched_ids

    final = pc.finalize_segment_timestamps(matched, total_duration=9999.0)
    assert_no_overlapping_clips(final)


def test_ellipsis_spans_with_coincident_start_no_overlap():
    """Reproduces the ch022 segment325/328 failure where seg N and seg N+3 share
    the same audio start (zero gap). Must not produce an overlap."""
    spec = [
        ("seg325", "What"),
        ("seg326", "."),
        ("seg327", "."),
        ("seg328", "what if he isn't home?"),
    ]
    words = [
        ("what", 755.680, 756.881),
        # continuation begins at the SAME start as seg325
        ("what", 755.680, 756.5),
        ("if", 756.6, 756.8),
        ("he", 756.9, 757.1),
        ("isnt", 757.2, 757.6),
        ("home", 757.7, 759.162),
    ]
    segments = make_segments(["seg325", "seg326", "seg327", "seg328"])
    matched = align(html_tokens_from(spec), audio_tokens_from(words), segments)
    matched_ids = {m["id"] for m in matched}
    assert "seg326" not in matched_ids and "seg327" not in matched_ids

    final = pc.finalize_segment_timestamps(matched, total_duration=9999.0)
    assert_no_overlapping_clips(final)


def test_ellipsis_spans_with_positive_gap_still_interpolated():
    """When there IS room (positive gap) the ellipsis spans are still interpolated
    and the result remains non-overlapping."""
    spec = [
        ("seg1", "the bad times"),
        ("seg2", "."),
        ("seg3", "."),
        ("seg4", "came back"),
    ]
    words = [
        ("the", 10.0, 10.2),
        ("bad", 10.3, 10.5),
        ("times", 10.6, 11.0),
        # clear pause, continuation starts well after seg1 ends
        ("came", 14.0, 14.3),
        ("back", 14.4, 14.8),
    ]
    segments = make_segments(["seg1", "seg2", "seg3", "seg4"])
    matched = align(html_tokens_from(spec), audio_tokens_from(words), segments)
    matched_ids = {m["id"] for m in matched}
    assert "seg2" in matched_ids and "seg3" in matched_ids  # interpolated

    final = pc.finalize_segment_timestamps(matched, total_duration=9999.0)
    assert_no_overlapping_clips(final)


# --- 4. regression: ordinary multi-word segments unaffected ----------------


def test_plain_segments_align_unchanged():
    spec = [
        ("seg1", "I took the aspirin he had put on the counter"),
        ("seg2", "Then I got up and walked slowly"),
    ]
    words = [
        ("I", 1.0, 1.1),
        ("took", 1.2, 1.4),
        ("the", 1.5, 1.6),
        ("aspirin", 1.7, 2.0),
        ("he", 2.1, 2.2),
        ("had", 2.3, 2.4),
        ("put", 2.5, 2.6),
        ("on", 2.7, 2.8),
        ("the", 2.9, 3.0),
        ("counter", 3.1, 3.5),
        ("then", 4.0, 4.2),
        ("i", 4.3, 4.4),
        ("got", 4.5, 4.6),
        ("up", 4.7, 4.8),
        ("and", 4.9, 5.0),
        ("walked", 5.1, 5.3),
        ("slowly", 5.4, 5.8),
    ]
    segments = make_segments(["seg1", "seg2"])
    matched = by_id(align(html_tokens_from(spec), audio_tokens_from(words), segments))
    assert matched["seg1"]["start"] == 1.0
    assert matched["seg1"]["end"] == 3.5  # 'counter'
    assert matched["seg2"]["start"] == 4.0  # 'then'
    assert matched["seg2"]["end"] == 5.8  # 'slowly'


def test_apostrophe_word_matches_asr():
    """'he'd' in HTML and ASR both normalize to 'hed' -> matches, no drop."""
    spec = [("seg1", "the aspirin he'd put down")]
    words = [
        ("the", 1.0, 1.1),
        ("aspirin", 1.2, 1.5),
        ("he'd", 1.6, 1.8),
        ("put", 1.9, 2.0),
        ("down", 2.1, 2.3),
    ]
    segments = make_segments(["seg1"])
    matched = by_id(align(html_tokens_from(spec), audio_tokens_from(words), segments))
    assert matched["seg1"]["start"] == 1.0
    assert matched["seg1"]["end"] == 2.3


def test_hyphen_compound_when_asr_keeps_it_whole():
    """If ASR happens to emit one hyphenated word, both sides split it the same
    way and the segment still matches."""
    spec = [
        ("seg1", "he left"),
        ("seg2", "dry-mouthed"),
        ("seg3", "again"),
    ]
    words = [
        ("he", 1.0, 1.1),
        ("left", 1.2, 1.4),
        ("dry-mouthed", 1.5, 2.0),  # single ASR word with a hyphen
        ("again", 2.1, 2.4),
    ]
    segments = make_segments(["seg1", "seg2", "seg3"])
    matched = by_id(align(html_tokens_from(spec), audio_tokens_from(words), segments))
    assert "seg2" in matched and not matched["seg2"].get("interpolated", False)
    assert matched["seg2"]["start"] == 1.5
    assert matched["seg2"]["end"] == 2.0


# --- 5. build_raw_matches direct behavior ---------------------------------


def test_envelope_is_min_start_max_end():
    html_tokens = html_tokens_from([("s", "alpha beta gamma")])
    audio_tokens = audio_tokens_from(
        [("alpha", 5.0, 5.5), ("beta", 6.0, 6.5), ("gamma", 7.0, 7.5)]
    )
    matcher = difflib.SequenceMatcher(
        None,
        [t["token"] for t in html_tokens],
        [t["token"] for t in audio_tokens],
        autojunk=False,
    )
    raw, *_ = pc.build_raw_matches(matcher.get_opcodes(), html_tokens, audio_tokens)
    assert raw["s"]["start"] == 5.0
    assert raw["s"]["end"] == 7.5


# --- 6. short divider-page matching (Option 2) -----------------------------


def test_build_match_windows_short_page_gets_one_full_window():
    # 7-token divider page ("11/22/63 PART 6 THE GREEN CARD MAN").
    tokens = [{"token": t} for t in ["112263", "part", "6", "the", "green", "card", "man"]]
    assert pc.build_match_windows(tokens) == [(0, 7)]


def test_build_match_windows_below_indexing_gate_is_empty():
    # part5.html has 4 tokens -> below the >=5 indexing gate -> no window.
    tokens = [{"token": t} for t in ["112263", "part", "5", "x"]]
    assert pc.build_match_windows(tokens) == []


def test_build_match_windows_exactly_min_tokens():
    assert pc.build_match_windows([{"token": "x"}] * 5) == [(0, 5)]


def test_build_match_windows_long_page_unchanged_multi_window():
    # A long page keeps the regular overlapping windows and gets no extra full-file
    # window appended.
    windows = pc.build_match_windows([{"token": "x"}] * 200)
    assert len(windows) == 2
    assert (0, 200) not in windows  # not collapsed to a single full-file window
    assert windows[0] == (0, pc.MATCH_WINDOW_SIZE)


def test_build_match_windows_just_below_window_size_single_window():
    # 159 tokens: one regular window (0,159), still >= MATCH_WINDOW_MIN_TOKENS.
    assert pc.build_match_windows([{"token": "x"}] * 159) == [(0, 159)]


def test_is_short_page_candidate():
    assert pc.is_short_page_candidate({"window_start": 0, "window_end": 7})
    assert not pc.is_short_page_candidate({"window_start": 0, "window_end": 160})
    assert not pc.is_short_page_candidate({"window_start": 80, "window_end": 200})
    # A full window equal to the minimum is not "short".
    assert not pc.is_short_page_candidate(
        {"window_start": 0, "window_end": pc.MATCH_WINDOW_MIN_TOKENS}
    )


# Real page/probe token streams from the book (chapter 9 audio chunk 030 + part6).
PART6_PAGE = ["112263", "part", "6", "the", "green", "card", "man"]
PROBE_030 = [
    "part", "6", "the", "green", "card", "man",
    "daily", "news", "extra", "saturday", "november", "23", "1963",
    "jfk", "escapes", "assassination",
]
PROBE_037 = ["11", "22", "63", "a", "novel", "was", "written", "by", "stephen", "king"]


def test_probe_contains_page_prefix_matches_after_header_skip():
    # The spoken probe omits the printed "11/22/63" running header but says the rest
    # of the heading consecutively -> match.
    assert pc.probe_contains_page_prefix(PROBE_030, PART6_PAGE)


def test_probe_contains_page_prefix_rejects_credits():
    assert not pc.probe_contains_page_prefix(PROBE_037, PART6_PAGE)


def test_probe_contains_page_prefix_rejects_other_dividers():
    # 030 must not match a different part divider.
    for page in (
        ["112263", "part", "1", "watershed", "moment"],
        ["112263", "part", "2", "the", "janitors", "father"],
        ["112263", "part", "3", "living", "in", "the", "past"],
        ["112263", "part", "4", "sadie", "and", "the", "general"],
        ["112263", "for", "zelda", "hey", "honey", "welcome", "to", "the", "party"],
    ):
        assert not pc.probe_contains_page_prefix(PROBE_030, page)


def test_probe_contains_page_prefix_tolerates_leading_stray_token():
    assert pc.probe_contains_page_prefix(["uh"] + PROBE_030, PART6_PAGE)


def test_probe_contains_page_prefix_rejects_far_offset():
    # Heading run starting too far into the probe is rejected.
    assert not pc.probe_contains_page_prefix(["x", "x", "x"] + PROBE_030, PART6_PAGE)


def test_probe_contains_page_prefix_requires_full_post_header_run():
    # The whole page (after skipping up to SHORT_PAGE_HEADER_SKIP leading header
    # tokens) must appear contiguously. A probe that diverges partway does not match.
    page = ["112263", "part", "6", "the", "green", "card", "man"]
    probe = ["part", "6", "the", "green", "card", "completely", "different"]
    assert not pc.probe_contains_page_prefix(probe, page)  # missing final "man"


def test_probe_contains_page_prefix_only_part_overlap_rejected():
    page = ["112263", "part", "6", "the", "green", "card", "man"]
    probe = ["part", "completely", "different", "words", "here", "now"]
    assert not pc.probe_contains_page_prefix(probe, page)


def test_probe_contains_page_prefix_empty_inputs():
    assert not pc.probe_contains_page_prefix([], PART6_PAGE)
    assert not pc.probe_contains_page_prefix(PROBE_030, [])


def test_probe_contains_page_prefix_header_skip_bounded():
    # Header skip is bounded (SHORT_PAGE_HEADER_SKIP) so a match can't be achieved by
    # discarding most of the page. Here only the last 2 tokens are spoken, which would
    # require skipping 5 leading tokens -> rejected.
    page = ["112263", "part", "6", "the", "green", "card", "man"]
    probe = ["card", "man", "and", "more"]
    assert not pc.probe_contains_page_prefix(probe, page)


def select_short_page_match(probe_tokens, candidates, last_matched_html_order):
    """Reimplementation of the short-page fallback decision in link_html_with_audio
    (the `else` branch): pick the first short-page candidate that is in forward order
    and whose heading is spoken at the probe start. Kept in sync with pipeline_core.

    Returns the chosen candidate dict or None.
    """
    for candidate in candidates:
        if not pc.is_short_page_candidate(candidate):
            continue
        if candidate["html_order"] < last_matched_html_order:
            continue
        if pc.probe_contains_page_prefix(probe_tokens, candidate["window_tokens"]):
            return candidate
    return None


def _part6_candidate():
    return {
        "file_name": "ops/xhtml/part6.html",
        "html_order": 46,
        "window_index": 0,
        "window_start": 0,
        "window_end": 7,
        "window_tokens": PART6_PAGE,
        "global_index": 1234,
    }


def test_short_page_match_accepts_030_in_forward_order():
    # Previous match was ch028 at html_order 45; part6 at 46 is forward -> accepted.
    chosen = select_short_page_match(PROBE_030, [_part6_candidate()], last_matched_html_order=45)
    assert chosen is not None
    assert chosen["file_name"] == "ops/xhtml/part6.html"


def test_short_page_match_rejected_when_backward_order():
    # A later chapter (e.g. ch029 at 47) was already matched -> part6 at 46 is a
    # backward jump and must be rejected by the hard forward-order gate.
    chosen = select_short_page_match(PROBE_030, [_part6_candidate()], last_matched_html_order=47)
    assert chosen is None


def test_short_page_match_rejected_for_credits():
    chosen = select_short_page_match(PROBE_037, [_part6_candidate()], last_matched_html_order=45)
    assert chosen is None


def test_short_page_match_ignores_non_short_candidates():
    long_cand = {
        "file_name": "ops/xhtml/ch028.html",
        "html_order": 45,
        "window_index": 0,
        "window_start": 0,
        "window_end": 160,
        "window_tokens": PART6_PAGE + ["x"] * 200,
        "global_index": 10,
    }
    chosen = select_short_page_match(PROBE_030, [long_cand], last_matched_html_order=0)
    assert chosen is None


# --- TOC exemption in unmatched-spine validation ---------------------------


def _page(text):
    return f"<html><body><p>{text}</p></body></html>"


def _write_epub(files):
    """Write an in-memory EPUB to a temp file and return its path."""
    handle = tempfile.NamedTemporaryFile(suffix=".epub", delete=False)
    handle.close()
    with zipfile.ZipFile(handle.name, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return handle.name


_LONG = "word " * 200  # comfortably above the 80-token substantive threshold


def _three_spine_opf(guide_toc=False, mid_nav=False):
    mid_props = ' properties="nav"' if mid_nav else ""
    guide = (
        '<guide><reference type="toc" href="mid.xhtml"/></guide>' if guide_toc else ""
    )
    return (
        '<?xml version="1.0"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0"><manifest>'
        '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
        f'<item id="mid" href="mid.xhtml" media-type="application/xhtml+xml"{mid_props}/>'
        '<item id="c2" href="c2.xhtml" media-type="application/xhtml+xml"/>'
        '</manifest>'
        '<spine><itemref idref="c1"/><itemref idref="mid"/><itemref idref="c2"/></spine>'
        f'{guide}</package>'
    )


def _run_unmatched_check(opf, mid_body):
    path = _write_epub(
        {
            "content.opf": opf,
            "c1.xhtml": _page(_LONG),
            "mid.xhtml": _page(mid_body),
            "c2.xhtml": _page(_LONG),
        }
    )
    try:
        book_info = {
            "matched_list": [{"html_file": "c1.xhtml"}, {"html_file": "c2.xhtml"}],
            "epub_file": path,
            "out_file": path,
            "folder_name": os.path.dirname(path),
        }
        return pc.test_unmatched_spine_html(book_info)
    finally:
        os.unlink(path)


def test_toc_like_files_detected_from_guide_and_nav():
    opf = _three_spine_opf(guide_toc=True, mid_nav=True)
    path = _write_epub({"content.opf": opf, "mid.xhtml": _page(_LONG)})
    try:
        with zipfile.ZipFile(path) as zf:
            toc_like = pc.get_toc_like_html_files(zf)
    finally:
        os.unlink(path)
    assert "mid.xhtml" in toc_like


def test_unmatched_spine_exempts_guide_toc_page():
    result = _run_unmatched_check(_three_spine_opf(guide_toc=True), _LONG)
    assert result["ok"] is True
    assert result["findings"] == []


def test_unmatched_spine_exempts_nav_property_page():
    result = _run_unmatched_check(_three_spine_opf(mid_nav=True), _LONG)
    assert result["ok"] is True
    assert result["findings"] == []


def test_unmatched_spine_exempts_in_document_nav():
    # No OPF signal; the page self-identifies with a <nav> element.
    body = "<nav><a href='#c1'>Chapter 1</a></nav>" + _LONG
    result = _run_unmatched_check(_three_spine_opf(), body)
    assert result["ok"] is True


def test_unmatched_spine_still_flags_real_missing_chapter():
    # Substantive, unmatched, and NOT a TOC/nav -> must still be reported.
    result = _run_unmatched_check(_three_spine_opf(), _LONG)
    assert result["ok"] is False
    assert [f["html_file"] for f in result["findings"]] == ["mid.xhtml"]


def test_html_declares_toc_helper():
    assert pc._html_declares_toc(BeautifulSoup("<nav>x</nav>", "lxml"))
    assert pc._html_declares_toc(
        BeautifulSoup('<div epub:type="toc">x</div>', "lxml")
    )
    assert not pc._html_declares_toc(BeautifulSoup("<p>plain</p>", "lxml"))


if __name__ == "__main__":
    from _runner import run_module_tests

    raise SystemExit(run_module_tests(globals()))
    sys.exit(1 if failures else 0)

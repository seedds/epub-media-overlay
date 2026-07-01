"""Segmentation tests for mark_sentence's sentence/phrase boundary detection.

These pin the read-aloud chunking produced by
`_get_sentence_aware_segment_boundaries` (Punkt boundaries + abbreviation/ellipsis
merge repairs + lone-initial split repair + quote/paren/dash phrase splitting +
short-fragment merging). Each case is a `(text, expected_segments)` pair; the test
asserts the exact list of segment substrings.

These previously lived in `mark_sentence.py`'s `__main__` block as manual
`test_your_case(...)` calls; they are now parametrized so each case reports
individually under pytest.

Run:
  /Users/f2pgod/Documents/spyder312/bin/python -m pytest tests/test_segmentation.py -q
or:
  /Users/f2pgod/Documents/spyder312/bin/python tests/test_segmentation.py
"""

# Bootstrap: allow running this file directly (`python tests/test_segmentation.py`).
# Source modules are imported by bare name below, so the repo root must be on
# sys.path before those imports. Under pytest, pyproject's pythonpath handles this.
import sys as _sys
from pathlib import Path as _Path

_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import pytest

from mark_sentence import _get_sentence_aware_segment_boundaries

LANGUAGE = "english"
MIN_WORDS = 1


# Each entry: (text, expected_segments). Kept 1:1 with the original manual cases.
CASES: list[tuple[str, list[str]]] = [
    (
        "the guy who passed along the tip gets a bigger allocation than normal.” The merchant tapped his fingers on the desk.",
        [
            "the guy who passed along the tip gets a bigger allocation than normal.” ",
            "The merchant tapped his fingers on the desk.",
        ],
    ),
    (
        "“This is the hedgehog named Mr. Needlemouse?”",
        ["“This is the hedgehog named Mr. Needlemouse?”"],
    ),
    (
        "The discussion was set to begin around 7:30 p.m. in Peter Main’s hotel suite at Caesar’s Palace.",
        [
            "The discussion was set to begin around 7:30 p.m. in Peter Main’s hotel suite at Caesar’s Palace."
        ],
    ),
    (
        "and in between you spend a bunch of years searching for it—looking cool,",
        [
            "and in between you spend a bunch of years searching for it—",
            "looking cool,",
        ],
    ),
    (
        "In the middle of Mr. Y.’s speech,",
        ["In the middle of Mr. Y.’s speech,"],
    ),
    (
        "“a meaningful moment in U.S. history.”",
        ["“a meaningful moment in U.S. history.”"],
    ),
    (
        "it’s us. Electronics is gonna replace banking in Dallas by 1970.",
        [
            "it’s us. ",
            "Electronics is gonna replace banking in Dallas by 1970.",
        ],
    ),
    (
        "Give it to us. We will handle it.",
        [
            "Give it to us. ",
            "We will handle it.",
        ],
    ),
    (
        "They left the U.S. in 1970 for good.",
        ["They left the U.S. in 1970 for good."],
    ),
    (
        "It happened in the U.S. The economy boomed.",
        ["It happened in the U.S. The economy boomed."],
    ),
    (
        "such years since I heard anything of you … must come to Soldier Island …",
        [
            "such years since I heard anything of you … ",
            "must come to Soldier Island …",
        ],
    ),
    (
        '"Hello," she said.',
        ['"Hello," ', "she said."],
    ),
    (
        "I could regale you with tales of how we had great fun on the trip, but I won’t. I don’t feel like reliving it right now.",
        [
            "I could regale you with tales of how we had great fun on the trip, ",
            "but I won’t. ",
            "I don’t feel like reliving it right now.",
        ],
    ),
    (
        "and Al was seriously ill. I could see that in a single glance.",
        [
            "and Al was seriously ill. ",
            "I could see that in a single glance.",
        ],
    ),
    (
        "but it might garner glances in a decade where shaving the back of the neck was considered a normal part of the barbering service and sideburns were reserved for rockabilly dudes like the one who had called me Daddy-O. Of course I could say I was a tourist,",
        [
            "but it might garner glances in a decade where shaving the back of the neck was considered a normal part of the barbering service and sideburns were reserved for rockabilly dudes like the one who had called me Daddy-O. ",
            "Of course I could say I was a tourist,",
        ],
    ),
    (
        "It cost 1,000 dollars, not 2,000.",
        [
            "It cost 1,000 dollars, ",
            "not 2,000.",
        ],
    ),
    (
        "At 3:45, we left.",
        [
            "At 3:45, ",
            "we left.",
        ],
    ),
    (
        "He ran (very fast) home.",
        [
            "He ran ",
            "(very fast) ",
            "home.",
        ],
    ),
    (
        "I was going to—wait, what?",
        [
            "I was going to—",
            "wait, ",
            "what?",
        ],
    ),
    (
        "Really?! I can't believe it.",
        [
            "Really?! ",
            "I can't believe it.",
        ],
    ),
    (
        "he had taken off his hat and his hair stood out around his head like that of a cartoon nebbish who has just inserted Finger A in Electric Socket B. He was gesticulating at the clerk with both hands, ",
        [
            "he had taken off his hat and his hair stood out around his head like that of a cartoon nebbish who has just inserted Finger A in Electric Socket B. ",
            "He was gesticulating at the clerk with both hands, ",
        ],
    ),
    (
        "George W. Bush went home.",
        ["George W. Bush went home."],
    ),
    (
        "I saw Mr. B. Smith arrive.",
        ["I saw Mr. B. Smith arrive."],
    ),
    (
        "It is option A. We chose it.",
        [
            "It is option A. ",
            "We chose it.",
        ],
    ),
    (
        "he had taken off his hat and his hair stood out around his head like that of a cartoon nebbish who has just inserted Finger A in Electric Socket B. He was gesticulating at the clerk with both hands,",
        [
            "he had taken off his hat and his hair stood out around his head like that of a cartoon nebbish who has just inserted Finger A in Electric Socket B. ",
            "He was gesticulating at the clerk with both hands,",
        ],
    ),
    (
        "she looked at me reproachfully . . .",
        ["she looked at me reproachfully . . ."],
    ),
    (
        "Well . . . I suppose so.",
        [
            "Well . . . ",
            "I suppose so.",
        ],
    ),
    (
        "“My degree’s from Oklahoma, but . . .”",
        [
            "“My degree’s from Oklahoma, ",
            "but . . .”",
        ],
    ),
    (
        "Neither did I. Some silences can be comfortable. ",
        [
            "Neither did I. ",
            "Some silences can be comfortable. ",
        ],
    ),
    (
        "So did I. Everyone agreed.",
        [
            "So did I. ",
            "Everyone agreed.",
        ],
    ),
    (
        "I saw Mr. I. Jones today.",
        ["I saw Mr. I. Jones today."],
    ),
    (
        "The report was written by Dr. I. Newman.",
        ["The report was written by Dr. I. Newman."],
    ),
]


def _segments(text: str) -> list[str]:
    boundaries = _get_sentence_aware_segment_boundaries(text, LANGUAGE, MIN_WORDS)
    return [text[start:end] for start, end in boundaries]


@pytest.mark.parametrize(
    "text, expected",
    CASES,
    ids=[text[:40] for text, _ in CASES],
)
def test_segmentation(text, expected):
    assert _segments(text) == expected


if __name__ == "__main__":
    # No-pytest fallback: parametrized cases can't be called arg-less, so iterate
    # CASES directly. Mirrors tests/_runner.py's PASS/FAIL reporting.
    failures = []
    for text, expected in CASES:
        label = repr(text)
        try:
            got = _segments(text)
            assert got == expected, f"expected {expected!r}, got {got!r}"
            print(f"PASS {label}")
        except AssertionError as exc:
            failures.append((label, exc))
            print(f"FAIL {label}: {exc}")
    print(f"\n{len(CASES) - len(failures)}/{len(CASES)} passed")
    raise SystemExit(1 if failures else 0)

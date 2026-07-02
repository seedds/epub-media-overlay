"""
Whole-EPUB reference index for safe, evidence-based tag cleanup.

Purpose
-------

`mark_sentence.preprocess_remove_redundant_tags()` removes markup and attributes
only when it can prove they are *unreferenced* anywhere in the book. This module
builds the evidence it needs: two sets naming every `class` and every `id` that is
referenced by something in the EPUB.

An attribute value is considered "referenced" (and therefore must be preserved) if
it appears as:

- a CSS class token (`.foo`) or id token (`#foo`) in any stylesheet or inline
  `<style>` block, including inside compound / descendant / grouped / pseudo /
  attribute selectors;
- an id used as a link/nav fragment target (`href`/`src` ending in `#foo`,
  including NCX `<content src="...#foo"/>`);
- a token or string literal in any JavaScript file or inline `<script>` (e.g.
  `getElementById('book-inner')`), which the CSS/link passes cannot see.

Design bias
-----------

The index deliberately *over-captures*. When in doubt (an exotic selector, a JS
string that might be an id or a class) the identifier is added to both sets. Keeping
a still-unused attribute is harmless; deleting a used one silently breaks styling,
navigation, or scripted behaviour. Every parse/read failure is swallowed and simply
contributes nothing, so a malformed stylesheet can never *narrow* the sets.
"""

from __future__ import annotations

import posixpath
import re
import warnings
import zipfile
from dataclasses import dataclass

import tinycss2
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Manifest media-types (and file extensions as a fallback) used to classify the
# entries we need to read.
_CSS_MEDIA_TYPES = {"text/css"}
_JS_MEDIA_TYPES = {
    "application/javascript",
    "text/javascript",
    "application/x-javascript",
}
_NCX_MEDIA_TYPES = {"application/x-dtbncx+xml"}
_HTML_MEDIA_TYPES = {"application/xhtml+xml", "text/html"}

# Identifier-like tokens in JavaScript. Matches ids/classes that scripts reference
# via getElementById('x'), querySelector('#x'), getElementsByClassName('y'), etc.
# Length >= 2 avoids flooding the sets with single-letter loop variables while still
# catching realistic identifiers like "book-inner".
_JS_TOKEN_RE = re.compile(r"[A-Za-z_][-A-Za-z0-9_]{1,}")

# CSS properties that have no rendering effect for standard reflowable, left-to-right
# EPUB content. A rule whose declarations are *entirely* drawn from this set does not
# actually style anything, so the classes/ids in its selector are treated as
# unreferenced (i.e. removable). This is what lets vendor no-op rules such as
# `.koboSpan { -webkit-text-combine: inherit; }` be cleaned away without changing
# appearance. Kept intentionally tiny and conservative: only add a property here when
# it is known to be inert for LTR reflow. `text-combine-upright` and its -webkit-
# alias only affect vertical (CJK tate-chu-yoko) layout.
_INERT_CSS_PROPERTIES = {
    "-webkit-text-combine",
    "text-combine-upright",
}


@dataclass(frozen=True)
class ReferenceIndex:
    """Sets of class/id names that are referenced somewhere in the EPUB."""

    referenced_classes: frozenset
    referenced_ids: frozenset


def _iter_manifest_sources(opf_soup, opf_dir):
    """Yield (resolved_zip_path, kind) for manifest items we care about.

    `kind` is one of "css", "js", "html", "ncx". Classification prefers the
    declared media-type and falls back to the file extension.
    """
    for item in opf_soup.find_all("item"):
        href = item.get("href")
        if not href:
            continue
        media_type = (item.get("media-type") or "").strip().lower()
        resolved = posixpath.normpath(posixpath.join(opf_dir, href))
        lower = resolved.lower()

        if media_type in _CSS_MEDIA_TYPES or lower.endswith(".css"):
            yield resolved, "css"
        elif media_type in _JS_MEDIA_TYPES or lower.endswith(".js"):
            yield resolved, "js"
        elif media_type in _NCX_MEDIA_TYPES or lower.endswith(".ncx"):
            yield resolved, "ncx"
        elif media_type in _HTML_MEDIA_TYPES or lower.endswith((".xhtml", ".html", ".htm")):
            yield resolved, "html"


def _read_zip_text(zip_file, name):
    """Read a zip member as text, or return None if it is missing/unreadable."""
    try:
        with zip_file.open(name) as handle:
            return handle.read().decode("utf-8", errors="replace")
    except (KeyError, OSError, zipfile.BadZipFile):
        return None


def _collect_css_tokens(css_text, classes, ids):
    """Add class/id names from a stylesheet's selectors to the given sets.

    Class tokens are `.` immediately followed by an ident; id tokens are `hash`
    tokens. Attribute-selector blocks (`[class~="x"]`) are descended into and any
    ident or string they contain is treated, conservatively, as both a class and an
    id name.

    A rule only contributes references if it actually styles something: rules whose
    declaration block is empty or made up entirely of inert properties (see
    `_INERT_CSS_PROPERTIES`) are skipped, so vendor no-op rules such as
    `.koboSpan { -webkit-text-combine: inherit; }` do not protect their selector.
    """
    try:
        rules = tinycss2.parse_stylesheet(
            css_text, skip_whitespace=True, skip_comments=True
        )
    except Exception:
        return

    _collect_rule_list_tokens(rules, classes, ids)


def _collect_rule_list_tokens(rules, classes, ids):
    """Collect selector tokens from a list of tinycss2 rules.

    Recurses into conditional group at-rules (`@media`, `@supports`, etc.) so that
    selectors styled only inside a conditional block still contribute references.
    Otherwise a class used solely in `@media screen { .highlight { ... } }` would be
    absent from the index and later stripped from the book.
    """
    for rule in rules:
        if rule.type == "qualified-rule":
            if _rule_has_meaningful_declaration(rule):
                _collect_selector_tokens(rule.prelude, classes, ids)
        elif rule.type == "at-rule" and rule.content is not None:
            try:
                nested = tinycss2.parse_rule_list(
                    rule.content, skip_whitespace=True, skip_comments=True
                )
            except Exception:
                continue
            _collect_rule_list_tokens(nested, classes, ids)


def _rule_has_meaningful_declaration(rule):
    """Return True if the rule declares at least one non-inert property.

    On any parse difficulty we fail safe by returning True (treat the rule as
    meaningful), so uncertainty never causes a referenced class/id to be dropped.
    """
    try:
        declarations = tinycss2.parse_declaration_list(
            rule.content, skip_whitespace=True, skip_comments=True
        )
    except Exception:
        return True

    for decl in declarations:
        if decl.type != "declaration":
            continue
        if decl.lower_name not in _INERT_CSS_PROPERTIES:
            return True
    # An empty block styles nothing, and an all-inert block has no visible effect;
    # in both cases the selector's classes/ids are not meaningfully referenced.
    return False


def _collect_selector_tokens(tokens, classes, ids):
    prev_was_dot = False
    for token in tokens:
        ttype = token.type
        if ttype == "hash":
            ids.add(token.value)
            prev_was_dot = False
        elif ttype == "literal" and token.value == ".":
            prev_was_dot = True
        elif ttype == "ident":
            if prev_was_dot:
                classes.add(token.value)
            prev_was_dot = False
        elif ttype == "[] block":
            # Attribute selector, e.g. [class~="ttf"] or [id="x"]. We cannot cheaply
            # tell class from id here, so add any ident/string to both sets.
            for inner in token.content:
                if inner.type in ("ident", "string"):
                    classes.add(inner.value)
                    ids.add(inner.value)
            prev_was_dot = False
        else:
            prev_was_dot = False


_FRAGMENT_RE = re.compile(r"#([^#\s\"']+)")


def _collect_fragment_ids(markup, ids):
    """Add every `#fragment` target found in href/src-style attributes to `ids`."""
    for match in _FRAGMENT_RE.finditer(markup):
        ids.add(match.group(1))


def _collect_js_tokens(js_text, classes, ids):
    """Add every identifier-like token in JavaScript to both sets (conservative)."""
    for match in _JS_TOKEN_RE.finditer(js_text):
        token = match.group(0)
        classes.add(token)
        ids.add(token)


# Attributes whose value is a (possibly space-separated) list of *bare* element ids,
# with no leading `#`. These are how accessibility relationships and table headers
# reference ids, so the ids they name must be treated as referenced or the attribute
# stripper leaves dangling references (aria-labelledby -> deleted id, etc.).
_ID_REF_ATTRS = ("aria-labelledby", "aria-describedby", "aria-owns", "headers")


def _collect_idref_attributes(soup, ids):
    """Add ids referenced via aria-*/headers idref lists and <label for>."""
    for tag in soup.find_all(True):
        for attr in _ID_REF_ATTRS:
            value = tag.get(attr)
            if not value:
                continue
            # bs4 may return a list (for known space-separated attrs) or a raw string;
            # join then split so both shapes yield the individual id tokens.
            text = " ".join(value) if isinstance(value, list) else value
            for piece in text.split():
                ids.add(piece)
        if tag.name == "label":
            target = tag.get("for")
            if target:
                ids.add(target)


def _collect_from_html(markup, classes, ids):
    """Pull references out of an HTML file: inline <style>, <script>, and fragments."""
    _collect_fragment_ids(markup, ids)
    try:
        # XHTML often carries an XML prolog; parsing it with the HTML parser is fine
        # for our purpose (extracting <style>/<script>), so silence the advisory.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(markup, "lxml")
    except Exception:
        return
    for style in soup.find_all("style"):
        _collect_css_tokens(style.get_text(), classes, ids)
    for script in soup.find_all("script"):
        text = script.get_text()
        if text:
            _collect_js_tokens(text, classes, ids)
    _collect_idref_attributes(soup, ids)


def build_reference_index(zip_file, opf_soup, opf_dir):
    """Scan an open EPUB zip and return a ReferenceIndex of referenced class/id names.

    `zip_file` is an open `zipfile.ZipFile`. `opf_soup` is the parsed OPF and
    `opf_dir` its directory inside the zip, used to resolve manifest hrefs.
    """
    classes = set()
    ids = set()

    for resolved, kind in _iter_manifest_sources(opf_soup, opf_dir):
        text = _read_zip_text(zip_file, resolved)
        if text is None:
            continue
        if kind == "css":
            _collect_css_tokens(text, classes, ids)
        elif kind == "js":
            _collect_js_tokens(text, classes, ids)
        elif kind == "html":
            _collect_from_html(text, classes, ids)
        elif kind == "ncx":
            _collect_fragment_ids(text, ids)

    return ReferenceIndex(
        referenced_classes=frozenset(classes),
        referenced_ids=frozenset(ids),
    )

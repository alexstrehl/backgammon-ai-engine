"""Guard against Markdown that renders wrongly on GitHub but looks fine in a diff.

A wrap put `+ self-play), and skips...` at the start of a line in
experiments/benchmarks/README.md. Markdown read the leading `+ ` as a bullet,
which opened a list; the 4-space-indented shell commands after it became list
continuation instead of a code block, so every `# comment` line inside them
rendered as an <h1>. Nothing in the diff looked wrong.
"""
import html
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
DOCS = sorted(REPO.glob("experiments/**/README.md")) + [REPO / "README.md"]


def _lines(p):
    return p.read_text().split("\n")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_no_accidental_list_opener(doc):
    """A list marker must not start a line that continues a paragraph."""
    lines = _lines(doc)
    bad = []
    for i, ln in enumerate(lines, 1):
        if not re.match(r"^[-+*] ", ln):
            continue
        # Walk back to the blank line that opens this block. If the block
        # already contains a list marker at column 0 we are inside a list and
        # this is a genuine item; if it is plain prose, the marker is a wrapped
        # word that will silently open a list and swallow what follows.
        block = []
        for j in range(i - 2, -1, -1):
            if not lines[j].strip():
                break
            block.append(lines[j])
        # A lead-in ending in ':' (or a finished sentence) legitimately
        # introduces a list; the bug is a marker landing mid-sentence.
        lead_in = block[0].rstrip().endswith((":", ".", "!", "?")) if block else False
        if block and not lead_in and not any(re.match(r"^[-+*] |^\d+[.)] ", b) for b in block):
            bad.append(f"{doc.name}:{i}: {ln[:70]!r} (continues: {block[0][-45:]!r})")
    assert not bad, "accidental list opener — wraps a paragraph into a list:\n" + "\n".join(bad)


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_no_hash_at_heading_indent(doc):
    """`#` indented 1-3 spaces is an ATX heading, not code."""
    bad = [f"{doc.name}:{i}: {ln[:70]!r}"
           for i, ln in enumerate(_lines(doc), 1) if re.match(r"^ {1,3}#", ln)]
    assert not bad, "`#` at 1-3 spaces renders as a heading:\n" + "\n".join(bad)


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_headings_are_real_sections(doc):
    """If markdown_it is available, assert no shell comment became a heading."""
    md = pytest.importorskip("markdown_it", reason="markdown_it not installed")
    rendered = md.MarkdownIt("commonmark").render(doc.read_text())
    # Rendering turns `code` into <code>code</code> and escapes &/</>, so
    # compare backtick-free, entity-unescaped text on both sides.
    headings = [html.unescape(re.sub(r"<[^>]+>", "", h))
                for h in re.findall(r"<h[1-6][^>]*>(.*?)</h[1-6]>", rendered, re.S)]
    declared = {ln.lstrip("# ").strip().replace("`", "")
                for ln in _lines(doc) if ln.startswith("#")}
    stray = [h for h in headings if h.strip() not in declared]
    assert not stray, f"{doc.name}: rendered headings not written as headings:\n" + "\n".join(stray)

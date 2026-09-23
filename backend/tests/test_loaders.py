from app.ingestion.loaders import Page, _clean, _strip_references, find_section, load_markdown


def test_hyphenation_across_linebreak_is_rejoined():
    assert "representation" in _clean("This is a repre-\nsentation of text")


def test_intentional_hyphen_is_preserved():
    assert "state-of-the-art" in _clean("a state-of-the-art model")


def test_nul_bytes_are_stripped():
    # Regression: Postgres rejects NUL in text columns, which failed 12 of 99
    # real arXiv PDFs outright.
    assert "\x00" not in _clean("bad\x00text")
    assert _clean("bad\x00text") == "badtext"


def test_other_control_bytes_are_stripped_but_structure_survives():
    cleaned = _clean("a\x07b\x1fc\n\nd\te")
    assert cleaned == "abc\n\nd e"


def test_blank_line_runs_collapse():
    assert _clean("a\n\n\n\n\nb") == "a\n\nb"


def test_references_stripped_from_back_half():
    pages = [
        Page(1, "Introduction text"),
        Page(2, "Method text"),
        Page(3, "Results text"),
        Page(4, "Conclusion.\n\nReferences\n\n[1] Someone et al."),
    ]
    kept = _strip_references(pages)
    assert len(kept) == 4
    assert "Someone et al" not in kept[-1].text
    assert "Conclusion." in kept[-1].text


def test_inline_mention_of_references_does_not_truncate():
    # Only a line that is nothing but the heading counts.
    pages = [Page(1, "Body"), Page(2, "We survey References and related work here")]
    kept = _strip_references(pages)
    assert len(kept) == 2
    assert "related work here" in kept[1].text


def test_references_heading_in_front_half_is_ignored():
    # A heading this early is far more likely to be a forward reference than
    # the actual bibliography of a long paper.
    pages = [Page(1, "Intro\n\nReferences\n\n[1] x"), Page(2, "B"), Page(3, "C"), Page(4, "D")]
    assert len(_strip_references(pages)) == 4


def test_section_detection():
    assert find_section("3.1 Experimental Setup\n\nWe train...") == "3.1 Experimental Setup"
    assert find_section("Abstract\n\nWe present...") == "Abstract"
    assert find_section("just some body text") is None


def test_markdown_title_from_heading(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# Retrieval Notes\n\nBody text here.", encoding="utf-8")
    doc = load_markdown(path)
    assert doc.title == "Retrieval Notes"
    assert doc.source_type == "markdown"
    assert doc.page_count == 1


def test_content_hash_is_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.md"
    a.write_text("# T\n\nsame body", encoding="utf-8")
    b = tmp_path / "b.md"
    b.write_text("# T\n\ndifferent body", encoding="utf-8")

    assert load_markdown(a).content_hash == load_markdown(a).content_hash
    assert load_markdown(a).content_hash != load_markdown(b).content_hash

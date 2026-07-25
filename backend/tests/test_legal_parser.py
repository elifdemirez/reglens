"""Tests for EUR-Lex structure parsing."""

from app.services.document_processing.legal_parser import (
    detect_document_kind,
    parse_pages,
)
from tests.conftest import IVDR_SAMPLE, MDR_SAMPLE


def test_detects_mdr(mdr_pages):
    kind, label = detect_document_kind(MDR_SAMPLE)
    assert kind == "mdr"
    assert label == "MDR"


def test_detects_ivdr(ivdr_pages):
    kind, label = detect_document_kind(IVDR_SAMPLE)
    assert kind == "ivdr"
    assert label == "IVDR"


def test_detects_generic_regulation():
    text = "REGULATION (EU) 2019/1020\nArticle 1\nScope\nArticle 2\nDefinitions\nArticle 3\nRules"
    kind, label = detect_document_kind(text)
    assert kind == "regulation"
    assert label == "2019/1020"


def test_plain_text_is_general():
    kind, label = detect_document_kind("Just some meeting notes about coffee machines.")
    assert kind == "general"
    assert label is None


def test_tracks_article_and_chapter(mdr_pages):
    result = parse_pages(mdr_pages)
    article_10 = [b for b in result.blocks if b.article_num == 10]
    assert article_10, "Article 10 blocks should be recovered"
    assert all(b.chapter == "Chapter II" for b in article_10)
    assert article_10[0].heading == "General obligations of manufacturers"


def test_paragraph_numbers_are_captured(mdr_pages):
    result = parse_pages(mdr_pages)
    paragraphs = {b.paragraph for b in result.blocks if b.article_num == 10}
    assert "1" in paragraphs
    assert "9" in paragraphs


def test_definitions_are_classified(mdr_pages):
    result = parse_pages(mdr_pages)
    definitions = [b for b in result.blocks if b.kind == "definition"]
    assert len(definitions) >= 3
    assert any("medical device" in b.text for b in definitions)


def test_definition_paragraph_marker_is_stored_bare(mdr_pages):
    """Regression: storing "(1)" here rendered citations as "Article 2((1))"."""
    result = parse_pages(mdr_pages)
    definitions = [b for b in result.blocks if b.kind == "definition"]
    assert definitions
    for block in definitions:
        assert block.paragraph is not None
        assert "(" not in block.paragraph and ")" not in block.paragraph


def test_obligations_are_classified(mdr_pages):
    result = parse_pages(mdr_pages)
    obligations = [b for b in result.blocks if b.kind == "obligation"]
    assert obligations, "Blocks containing 'shall' should be tagged as obligations"


def test_lettered_points_become_list_blocks(mdr_pages):
    result = parse_pages(mdr_pages)
    assert any(b.kind == "list" for b in result.blocks)


def test_annex_is_tracked(mdr_pages):
    result = parse_pages(mdr_pages)
    annex_blocks = [b for b in result.blocks if b.annex == "Annex I"]
    assert annex_blocks
    assert annex_blocks[0].kind == "annex"


def test_section_path_is_readable(mdr_pages):
    result = parse_pages(mdr_pages)
    block = next(b for b in result.blocks if b.article_num == 10)
    assert block.section_path == "Chapter II > Article 10"


def test_page_furniture_is_dropped(mdr_pages):
    result = parse_pages(mdr_pages)
    joined = "\n".join(b.text for b in result.blocks)
    assert "Official Journal of the European Union" not in joined


def test_article_count(mdr_pages):
    result = parse_pages(mdr_pages)
    assert result.article_count == 4  # Articles 1, 2, 10, 13


def test_empty_input_is_safe():
    result = parse_pages([])
    assert result.blocks == []
    assert result.detected_kind == "general"


def test_paragraph_marker_alone_on_its_line_is_captured():
    """Real EUR-Lex PDFs break the marker onto its own line.

    Regression test: before this was handled, ~80% of chunks from the real MDR
    and IVDR carried no paragraph number, so citations lost their precision.
    """
    pages = [(1, "Article 10\nGeneral obligations\n1.\nManufacturers shall ensure conformity.\n"
                 "2.\nManufacturers shall keep technical documentation.\n")]
    result = parse_pages(pages)
    paragraphs = [b.paragraph for b in result.blocks if b.article_num == 10 and b.paragraph]
    assert "1" in paragraphs
    assert "2" in paragraphs


def test_bare_page_number_is_not_read_as_a_paragraph():
    pages = [(1, "Article 5\nScope\n42\nSome body text that follows the page number.\n")]
    result = parse_pages(pages)
    body = [b for b in result.blocks if b.article_num == 5]
    assert body
    assert body[0].paragraph is None

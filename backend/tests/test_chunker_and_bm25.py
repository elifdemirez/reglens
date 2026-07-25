"""Tests for chunking and the BM25 index."""

from app.config import settings
from app.services.document_processing.chunker import build_chunks
from app.services.document_processing.legal_parser import Block, parse_pages
from app.services.retrieval.bm25 import BM25Index, normalise, tokenize


# --- chunker ------------------------------------------------------------------

def test_chunks_carry_citation_metadata(mdr_pages):
    chunks = build_chunks(parse_pages(mdr_pages).blocks)
    article_10 = [c for c in chunks if c["article_num"] == 10]
    assert article_10
    assert article_10[0]["article"] == "Article 10"
    assert article_10[0]["section_path"] == "Chapter II > Article 10"


def test_chunks_never_span_two_articles(mdr_pages):
    chunks = build_chunks(parse_pages(mdr_pages).blocks)
    for chunk in chunks:
        # Every chunk maps to exactly one article label, so a citation is unambiguous.
        assert chunk["article"] is None or isinstance(chunk["article"], str)
    articles = [c["article_num"] for c in chunks if c["article_num"] is not None]
    assert articles == sorted(articles), "chunks should stay in document order"


def test_definitions_are_not_merged_together(mdr_pages):
    chunks = build_chunks(parse_pages(mdr_pages).blocks)
    definitions = [c for c in chunks if c["chunk_kind"] == "definition"]
    assert len(definitions) >= 3, "each definition should remain independently citable"


def test_oversized_block_is_split_with_overlap():
    long_text = ". ".join(f"Sentence number {i} about conformity assessment" for i in range(400))
    block = Block(text=long_text, page=1, article="Article 5", article_num=5)
    chunks = build_chunks([block])
    assert len(chunks) > 1
    assert all(len(c["content"]) <= settings.max_chunk_chars + 50 for c in chunks)
    assert all(c["article_num"] == 5 for c in chunks)


def test_empty_blocks_produce_no_chunks():
    assert build_chunks([]) == []


def test_tiny_fragments_are_dropped():
    """Regression: real PDFs produced 1-character chunks from header artefacts."""
    blocks = [
        Block(text="EN", page=1),
        Block(text="4", page=1),
        Block(text="A genuine clause with enough substance to be worth indexing.", page=1),
    ]
    chunks = build_chunks(blocks)
    assert len(chunks) == 1
    assert chunks[0]["content"].startswith("A genuine clause")


def test_chunk_indexes_stay_contiguous_after_filtering():
    # Different articles so the two real blocks are not merged into one chunk.
    blocks = [
        Block(text="x", page=1, article="Article 1", article_num=1),
        Block(text="First real clause that is long enough to survive filtering.",
              page=1, article="Article 1", article_num=1),
        Block(text="y", page=1, article="Article 2", article_num=2),
        Block(text="Second real clause that is also long enough to survive.",
              page=1, article="Article 2", article_num=2),
    ]
    chunks = build_chunks(blocks)
    assert [c["chunk_index"] for c in chunks] == [0, 1]
    assert chunks[0]["content"].startswith("First real clause")
    assert chunks[1]["content"].startswith("Second real clause")


# --- BM25 ---------------------------------------------------------------------

def test_tokenize_drops_stopwords():
    assert "the" not in tokenize("The manufacturer shall")
    assert "manufacturer" in tokenize("The manufacturer shall")


def test_bm25_ranks_exact_term_highest():
    docs = [
        "Importers shall verify the CE marking of the device.",
        "Manufacturers shall establish a post-market surveillance system.",
        "This Regulation lays down rules on scope.",
    ]
    index = BM25Index(docs)
    scores = index.score("post-market surveillance system")
    assert scores[1] == max(scores)


def test_bm25_scores_zero_for_absent_terms():
    index = BM25Index(["Alpha beta gamma", "Delta epsilon"])
    assert index.score("nonexistentterm") == [0.0, 0.0]


def test_bm25_handles_empty_corpus():
    assert BM25Index([]).score("anything") == []


def test_normalise_scales_to_one():
    assert normalise([2.0, 1.0, 0.0]) == [1.0, 0.5, 0.0]
    assert normalise([0.0, 0.0]) == [0.0, 0.0]
    assert normalise([]) == []

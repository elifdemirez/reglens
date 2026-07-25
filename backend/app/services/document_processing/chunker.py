"""Group parsed blocks into embedding-sized chunks.

Blocks are merged only while they share the same article, so a chunk never
straddles a legal boundary — that would make its citation ambiguous. Oversized
single blocks (long annex tables, mostly) are split with a character overlap so
a sentence cut in half still appears whole in one of the pieces.
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.services.document_processing.legal_parser import Block

# Real EUR-Lex PDFs yield stray one- and two-character fragments from running
# headers ("EN", a page number, a lone paragraph marker). They cost an embedding
# call and, worse, get glued onto the front of the next real clause during the
# merge step. Dropped *before* merging for that reason.
#
# Kept deliberately low: short annex entries like "Class IIa devices" are real
# content, so the threshold only has to clear obvious furniture.
MIN_BLOCK_CHARS = 15


def _split_oversized(block: Block, max_chars: int, overlap: int) -> list[Block]:
    text = block.text
    if len(text) <= max_chars:
        return [block]

    pieces: list[Block] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Prefer to break at a paragraph, then a sentence, then anywhere.
            window = text[start:end]
            for separator in ("\n\n", "\n", ". "):
                cut = window.rfind(separator)
                if cut > max_chars // 2:
                    end = start + cut + len(separator)
                    break
        piece_text = text[start:end].strip()
        if piece_text:
            pieces.append(
                Block(
                    text=piece_text,
                    page=block.page,
                    chapter=block.chapter,
                    section=block.section,
                    article=block.article,
                    article_num=block.article_num,
                    paragraph=block.paragraph,
                    annex=block.annex,
                    heading=block.heading,
                    kind=block.kind,
                )
            )
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def _same_article(a: Block, b: Block) -> bool:
    return a.article_num == b.article_num and a.annex == b.annex


def build_chunks(blocks: list[Block]) -> list[dict[str, Any]]:
    max_chars = settings.max_chunk_chars
    overlap = settings.chunk_overlap_chars

    normalised: list[Block] = []
    for block in blocks:
        if len(block.text.strip()) < MIN_BLOCK_CHARS:
            continue
        normalised.extend(_split_oversized(block, max_chars, overlap))

    merged: list[Block] = []
    for block in normalised:
        if (
            merged
            and _same_article(merged[-1], block)
            # Definitions stay standalone: each one is independently quotable.
            and merged[-1].kind != "definition"
            and block.kind != "definition"
            and len(merged[-1].text) + len(block.text) + 1 <= max_chars
        ):
            previous = merged[-1]
            previous.text = f"{previous.text}\n{block.text}"
            if previous.kind == "body" and block.kind != "body":
                previous.kind = block.kind
        else:
            merged.append(block)

    chunks: list[dict[str, Any]] = []
    for block in merged:
        if not block.text.strip():
            continue
        chunks.append(
            {
                # Indexed after filtering so chunk_index stays contiguous.
                "chunk_index": len(chunks),
                "content": block.text,
                "page": block.page,
                "article": block.article,
                "article_num": block.article_num,
                "paragraph": block.paragraph,
                "section_path": block.section_path or None,
                "heading": block.heading,
                "chunk_kind": block.kind,
                "embedding": None,
            }
        )
    return chunks

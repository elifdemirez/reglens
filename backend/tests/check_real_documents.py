"""Diagnostic: run the parser over the real MDR/IVDR PDFs and report what it found.

Not a pytest test — the PDFs are not in version control. Run manually:

    python tests/check_real_documents.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.document_processing.chunker import build_chunks  # noqa: E402
from app.services.document_processing.legal_parser import parse_pages  # noqa: E402
from app.services.document_processing.readers import read_document  # noqa: E402

SAMPLES = BACKEND_ROOT / "data" / "samples"


def report(path: Path) -> None:
    print("=" * 72)
    print(path.name)
    print("=" * 72)

    pages = read_document(path)
    print(f"pages extracted      : {len(pages)}")

    parsed = parse_pages(pages)
    print(f"detected kind        : {parsed.detected_kind}  (label={parsed.short_label})")
    print(f"articles found       : {parsed.article_count}")
    print(f"blocks               : {len(parsed.blocks)}")

    kinds = Counter(b.kind for b in parsed.blocks)
    print(f"block kinds          : {dict(kinds)}")

    chapters = sorted({b.chapter for b in parsed.blocks if b.chapter})
    print(f"chapters             : {len(chapters)} -> {chapters[:6]}{'…' if len(chapters) > 6 else ''}")

    annexes = sorted({b.annex for b in parsed.blocks if b.annex})
    print(f"annexes              : {len(annexes)} -> {annexes[:6]}{'…' if len(annexes) > 6 else ''}")

    chunks = build_chunks(parsed.blocks)
    print(f"chunks               : {len(chunks)}")
    sized = [len(c["content"]) for c in chunks]
    if sized:
        print(f"chunk chars min/avg/max : {min(sized)}/{sum(sized)//len(sized)}/{max(sized)}")

    with_article = sum(1 for c in chunks if c["article_num"] is not None)
    print(f"chunks with article  : {with_article} ({with_article * 100 // max(len(chunks), 1)}%)")
    with_paragraph = sum(1 for c in chunks if c["paragraph"])
    print(f"chunks with paragraph: {with_paragraph} ({with_paragraph * 100 // max(len(chunks), 1)}%)")

    # Spot-check a well-known clause: MDR/IVDR Article 10 binds manufacturers.
    article_10 = [c for c in chunks if c["article_num"] == 10]
    print(f"\nArticle 10 chunks    : {len(article_10)}")
    if article_10:
        sample = article_10[0]
        print(f"  section_path : {sample['section_path']}")
        print(f"  heading      : {sample['heading']}")
        print(f"  page         : {sample['page']}")
        print(f"  paragraphs   : {[c['paragraph'] for c in article_10]}")
        print(f"  kind         : {sample['chunk_kind']}")
        print(f"  text         : {sample['content'][:220]!r}…")

    definitions = [c for c in chunks if c["chunk_kind"] == "definition"]
    print(f"\ndefinition chunks    : {len(definitions)}")
    if definitions:
        print(f"  sample       : {definitions[0]['content'][:220]!r}…")
    print()


if __name__ == "__main__":
    files = sorted(SAMPLES.glob("*.pdf"))
    if not files:
        print(f"No PDFs found in {SAMPLES}")
        raise SystemExit(1)
    for pdf in files:
        report(pdf)

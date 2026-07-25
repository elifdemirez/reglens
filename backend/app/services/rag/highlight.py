"""Link answer sentences back to the source sentences that support them.

The UI uses this to highlight, inside each cited excerpt, the exact lines the
answer leaned on. It is a lexical overlap measure rather than a second model
call: it has to run after every answer, and on a CPU-only machine a second
inference pass would double an already slow response.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.retrieval.bm25 import tokenize

SENTENCE_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\d])|\n+")
MIN_OVERLAP = 0.34
MAX_SPANS_PER_SOURCE = 3


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Split into sentences, keeping each one's offsets in the original string."""
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for piece in SENTENCE_SPLIT.split(text):
        if piece is None:
            continue
        stripped = piece.strip()
        if not stripped:
            continue
        start = text.find(stripped, cursor)
        if start == -1:
            continue
        end = start + len(stripped)
        cursor = end
        spans.append((start, end, stripped))
    return spans


def _overlap(answer_tokens: set[str], sentence: str) -> float:
    tokens = set(tokenize(sentence))
    if not tokens or not answer_tokens:
        return 0.0
    shared = tokens & answer_tokens
    # Normalised against the sentence so a long clause is not rewarded just for
    # being long, and short boilerplate cannot score a spurious 1.0.
    return len(shared) / max(len(tokens), 6)


def find_supporting_spans(answer: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate each source with the character spans that support the answer."""
    answer_tokens = set(tokenize(answer))
    annotated: list[dict[str, Any]] = []

    for source in sources:
        content = source.get("content", "")
        scored: list[tuple[float, int, int]] = []
        for start, end, sentence in split_sentences(content):
            score = _overlap(answer_tokens, sentence)
            if score >= MIN_OVERLAP:
                scored.append((score, start, end))

        scored.sort(key=lambda item: item[0], reverse=True)
        spans = sorted(
            [{"start": s, "end": e, "score": round(sc, 3)} for sc, s, e in scored[:MAX_SPANS_PER_SOURCE]],
            key=lambda span: span["start"],
        )

        enriched = dict(source)
        enriched["highlights"] = spans
        annotated.append(enriched)

    return annotated

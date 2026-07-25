"""Post-generation checks on the model's answer.

None of these can prove an answer is correct. What they can do is catch the
failure modes a small local model actually exhibits: citing a document that was
never retrieved, answering at length from a context that clearly did not cover
the question, or degenerating into repetition.
"""

from __future__ import annotations

import re
from typing import Any

NO_ANSWER_TEXT = "The provided documents do not contain this information."

RE_CITATION = re.compile(r"\[([^\]]+)\]")


def extract_citations(answer: str) -> list[str]:
    return [c.strip() for c in RE_CITATION.findall(answer)]


# Phrasings the model reaches for when the context does not cover the question.
# Detecting them lets the answer be reported as a refusal rather than as a
# successful synthesis, which is what the UI badge and the history record show.
_REFUSAL_MARKERS = (
    "do not contain this information",
    "does not contain this information",
    "do not contain enough information",
    "does not contain enough information",
    "not contain any information",
    "no information about",
    "cannot answer",
    "unable to answer",
)


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def has_unsupported_citation(answer: str, sources: list[dict[str, Any]]) -> bool:
    """True when the answer cites a label that was not among the retrieved sources."""
    known_labels = set()
    for source in sources:
        citation = source.get("citation", "")
        known_labels.add(citation.lower())
        label = source.get("short_label") or source.get("filename") or ""
        if label:
            known_labels.add(label.lower())

    for citation in extract_citations(answer):
        lowered = citation.lower()
        if any(known.startswith(lowered[:12]) or lowered.startswith(known[:12])
               for known in known_labels if known):
            continue
        return True
    return False


def is_degenerate(answer: str) -> bool:
    """Detect the repetition loop small models fall into on weak context."""
    sentences = [s.strip().lower() for s in re.split(r"[.\n]+", answer) if len(s.strip()) > 25]
    if len(sentences) < 4:
        return False
    unique = len(set(sentences))
    return unique / len(sentences) < 0.5


def validate_answer(
    answer: str, sources: list[dict[str, Any]], confidence: float
) -> tuple[str, list[str]]:
    """Return the (possibly replaced) answer plus any warnings for the UI."""
    warnings: list[str] = []
    stripped = answer.strip()

    if not stripped:
        return NO_ANSWER_TEXT, ["The model returned an empty answer."]

    if is_degenerate(stripped):
        return (
            NO_ANSWER_TEXT,
            ["The model produced a repetitive answer, which usually means the retrieved "
             "context did not cover the question."],
        )

    if has_unsupported_citation(stripped, sources):
        warnings.append(
            "The answer cites a source that was not among the retrieved excerpts; "
            "treat the citation with caution."
        )

    if confidence < 0.35 and NO_ANSWER_TEXT.lower() not in stripped.lower():
        warnings.append(
            "Retrieval confidence is low — the excerpts may not fully cover this question."
        )

    return stripped, warnings

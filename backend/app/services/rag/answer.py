"""Answer orchestration: plan -> retrieve -> expand -> answer -> validate.

Two answer paths exist on purpose:

* **direct** — a definition or an enumerated list is already stated verbatim in
  the regulation. Paraphrasing it through a small model adds latency and risks
  distorting legal wording, so the excerpt is returned as-is.
* **synthesis** — everything else goes to the local chat model with the
  retrieved context.

On a CPU-only machine the direct path is the difference between an instant
answer and a ~30 second one, so it is taken whenever it is defensible.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from app.services.rag import prompts
from app.services.rag.highlight import find_supporting_spans
from app.services.rag.validate import NO_ANSWER_TEXT, is_refusal, validate_answer
from app.services.retrieval.expansion import build_context, format_context
from app.services.retrieval.hybrid import (
    compute_confidence,
    retrieve,
    retrieve_per_document_kind,
)
from app.services.retrieval.question_planner import QuestionPlan, plan_question

# Below this, the corpus almost certainly does not answer the question, so the
# chat model is never invoked. Calibrated against the real MDR/IVDR corpus: with
# real embeddings, on-topic questions scored 0.55-0.76 while an out-of-scope one
# ("tyre pressure for agricultural tractors") still reached 0.25 — an earlier
# threshold of 0.22 let it through and spent 48 s having the model refuse.
MIN_ANSWERABLE_CONFIDENCE = 0.35
# Above this, a definition/list chunk is trustworthy enough to return verbatim.
DIRECT_ANSWER_CONFIDENCE = 0.55


@dataclass
class AnswerResult:
    answer: str
    sources: list[dict[str, Any]]
    confidence: float
    answer_path: str          # direct | synthesis | refused
    question_type: str
    mode: str = "ask"
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "answer_path": self.answer_path,
            "question_type": self.question_type,
            "mode": self.mode,
            "warnings": self.warnings,
            "elapsed_ms": self.elapsed_ms,
        }


def _refusal(plan: QuestionPlan, sources: list[dict[str, Any]], confidence: float,
             started: float) -> AnswerResult:
    return AnswerResult(
        answer=NO_ANSWER_TEXT,
        sources=sources,
        confidence=confidence,
        answer_path="refused",
        question_type=plan.question_type,
        warnings=["No sufficiently relevant passage was found in the indexed documents."],
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


def _try_direct_answer(plan: QuestionPlan, results, confidence: float) -> str | None:
    """Return a verbatim excerpt when the question type allows it."""
    if not plan.allows_direct_answer or not results:
        return None
    if confidence < DIRECT_ANSWER_CONFIDENCE:
        return None

    top = results[0]
    if plan.question_type == "definition" and top.chunk_kind == "definition":
        return f"{top.content}\n\n[{top.citation}]"
    if plan.question_type == "list" and top.chunk_kind == "list":
        return f"{top.content}\n\n[{top.citation}]"
    return None


def answer_question(
    conn: sqlite3.Connection,
    question: str,
    models,
    *,
    document_ids: list[int] | None = None,
) -> AnswerResult:
    started = time.perf_counter()
    plan = plan_question(question)

    query_embedding = models.embed(question)

    if plan.is_comparison:
        return _answer_comparison(conn, plan, query_embedding, models, started)

    results = retrieve(conn, plan, query_embedding, document_ids=document_ids)
    confidence = compute_confidence(results, plan)
    sources = [chunk.to_dict() for chunk in results]

    if not results or confidence < MIN_ANSWERABLE_CONFIDENCE:
        return _refusal(plan, sources, confidence, started)

    direct = _try_direct_answer(plan, results, confidence)
    if direct is not None:
        return AnswerResult(
            answer=direct,
            sources=find_supporting_spans(direct, sources),
            confidence=confidence,
            answer_path="direct",
            question_type=plan.question_type,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    entries = build_context(conn, results)
    context = format_context(entries)
    user_prompt = f"Context:\n{context}\n\nQuestion: {plan.question}"
    raw = models.chat(prompts.prompt_for(plan.question_type), user_prompt)

    answer, warnings = validate_answer(raw, sources, confidence)
    return AnswerResult(
        answer=answer,
        sources=find_supporting_spans(answer, sources),
        confidence=confidence,
        # The model may decline even when retrieval looked adequate; reporting
        # that as a successful synthesis would misrepresent what happened.
        answer_path="refused" if is_refusal(answer) else "synthesis",
        question_type=plan.question_type,
        warnings=warnings,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


def _answer_comparison(
    conn: sqlite3.Connection,
    plan: QuestionPlan,
    query_embedding: list[float],
    models,
    started: float,
) -> AnswerResult:
    kinds = list(plan.doc_kinds) or ["mdr", "ivdr"]
    grouped = retrieve_per_document_kind(conn, plan, query_embedding, kinds)

    all_results = [chunk for chunks in grouped.values() for chunk in chunks]
    all_results.sort(key=lambda c: c.score, reverse=True)
    sources = [chunk.to_dict() for chunk in all_results]
    confidence = compute_confidence(all_results, plan)

    covered = [kind for kind, chunks in grouped.items() if chunks]
    if len(covered) < 2:
        missing = [k for k in kinds if k not in covered]
        return AnswerResult(
            answer=(
                "A comparison needs both regulations to be indexed and relevant. "
                f"No relevant passages were found for: {', '.join(m.upper() for m in missing)}."
            ),
            sources=sources,
            confidence=confidence,
            answer_path="refused",
            question_type="comparison",
            mode="compare",
            warnings=["Upload both regulations to enable comparison mode."],
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    sections = []
    for kind, chunks in grouped.items():
        if not chunks:
            continue
        label = chunks[0].short_label or kind.upper()
        body = format_context(build_context(conn, chunks, expand=False, max_chars=3000))
        sections.append(f"=== {label} ===\n{body}")
    context = "\n\n".join(sections)

    user_prompt = f"Context:\n{context}\n\nComparison question: {plan.question}"
    raw = models.chat(prompts.COMPARISON_PROMPT, user_prompt)

    answer, warnings = validate_answer(raw, sources, confidence)
    return AnswerResult(
        answer=answer,
        sources=find_supporting_spans(answer, sources),
        confidence=confidence,
        answer_path="refused" if is_refusal(answer) else "synthesis",
        question_type="comparison",
        mode="compare",
        warnings=warnings,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


def answer_question_stream(
    conn: sqlite3.Connection,
    question: str,
    models,
    *,
    document_ids: list[int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Same pipeline as :func:`answer_question`, emitted as events.

    Sources are sent before the answer text so the UI can render the evidence
    panel while the model is still generating.
    """
    started = time.perf_counter()
    plan = plan_question(question)
    yield {"type": "status", "stage": "planning", "question_type": plan.question_type}

    query_embedding = models.embed(question)
    yield {"type": "status", "stage": "retrieving"}

    if plan.is_comparison:
        result = _answer_comparison(conn, plan, query_embedding, models, started)
        yield {"type": "sources", "sources": result.sources, "confidence": result.confidence}
        yield {"type": "token", "text": result.answer}
        yield {"type": "done", **result.to_dict()}
        return

    results = retrieve(conn, plan, query_embedding, document_ids=document_ids)
    confidence = compute_confidence(results, plan)
    sources = [chunk.to_dict() for chunk in results]
    yield {"type": "sources", "sources": sources, "confidence": confidence}

    if not results or confidence < MIN_ANSWERABLE_CONFIDENCE:
        result = _refusal(plan, sources, confidence, started)
        yield {"type": "token", "text": result.answer}
        yield {"type": "done", **result.to_dict()}
        return

    direct = _try_direct_answer(plan, results, confidence)
    if direct is not None:
        result = AnswerResult(
            answer=direct,
            sources=find_supporting_spans(direct, sources),
            confidence=confidence,
            answer_path="direct",
            question_type=plan.question_type,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        yield {"type": "token", "text": direct}
        yield {"type": "done", **result.to_dict()}
        return

    entries = build_context(conn, results)
    context = format_context(entries)
    user_prompt = f"Context:\n{context}\n\nQuestion: {plan.question}"
    yield {"type": "status", "stage": "generating"}

    collected: list[str] = []
    for fragment in models.chat_stream(prompts.prompt_for(plan.question_type), user_prompt):
        collected.append(fragment)
        yield {"type": "token", "text": fragment}

    raw = "".join(collected)
    answer, warnings = validate_answer(raw, sources, confidence)
    result = AnswerResult(
        answer=answer,
        sources=find_supporting_spans(answer, sources),
        confidence=confidence,
        answer_path="refused" if is_refusal(answer) else "synthesis",
        question_type=plan.question_type,
        warnings=warnings,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
    # The validator may have replaced a degenerate answer after streaming it.
    yield {"type": "done", "replaced": answer != raw.strip(), **result.to_dict()}

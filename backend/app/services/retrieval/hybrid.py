"""Hybrid retrieval: semantic similarity + BM25 + legal structure signals."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.repositories import chunks_repo
from app.services.retrieval.bm25 import BM25Index, normalise
from app.services.retrieval.question_planner import QuestionPlan


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    filename: str
    short_label: str | None
    doc_kind: str
    content: str
    page: int | None
    article: str | None
    article_num: int | None
    paragraph: str | None
    section_path: str | None
    heading: str | None
    chunk_kind: str
    score: float
    semantic_score: float
    keyword_score: float
    structure_score: float

    @property
    def citation(self) -> str:
        label = self.short_label or self.filename
        parts = [label]
        if self.article:
            parts.append(
                f"{self.article}({self.paragraph})" if self.paragraph else self.article
            )
        elif self.section_path:
            parts.append(self.section_path)
        if self.page:
            parts.append(f"p. {self.page}")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "short_label": self.short_label,
            "doc_kind": self.doc_kind,
            "content": self.content,
            "page": self.page,
            "article": self.article,
            "paragraph": self.paragraph,
            "section_path": self.section_path,
            "heading": self.heading,
            "chunk_kind": self.chunk_kind,
            "citation": self.citation,
            "score": round(self.score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "structure_score": round(self.structure_score, 4),
        }


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Naming a specific article or annex is an explicit instruction, not a hint, so
# it is applied on top of the weighted blend rather than inside it — at a 0.1
# structure weight a soft bonus would lose to any chunk with better wording.
EXPLICIT_REFERENCE_BONUS = 0.5


def _structure_score(row: dict[str, Any], plan: QuestionPlan) -> float:
    """Reward chunks whose legal position matches what the question asked for.

    Capped at 1.0 so a structurally perfect but semantically irrelevant chunk
    still cannot outrank genuine content matches on its own.
    """
    score = 0.0

    if plan.preferred_chunk_kinds and row.get("chunk_kind") in plan.preferred_chunk_kinds:
        score += 0.5

    # Quoted terms of art should appear verbatim in the chunk.
    if plan.quoted_terms:
        content = (row.get("content") or "").lower()
        if any(term.lower() in content for term in plan.quoted_terms):
            score += 0.5

    return min(score, 1.0)


def _explicit_reference_bonus(row: dict[str, Any], plan: QuestionPlan) -> float:
    """Flat bonus for chunks the question named outright."""
    if plan.article_refs and row.get("article_num") in plan.article_refs:
        return EXPLICIT_REFERENCE_BONUS
    if plan.annex_refs:
        section_path = row.get("section_path") or ""
        if any(ref in section_path for ref in plan.annex_refs):
            return EXPLICIT_REFERENCE_BONUS
    return 0.0


def retrieve(
    conn: sqlite3.Connection,
    plan: QuestionPlan,
    query_embedding: list[float],
    *,
    top_k: int | None = None,
    document_ids: list[int] | None = None,
    strict_scope: bool = False,
) -> list[RetrievedChunk]:
    top_k = top_k or settings.top_k
    rows = chunks_repo.fetch_for_retrieval(conn, document_ids)
    if not rows:
        return []

    # Narrow to the regulations the question named. Normally an empty scope falls
    # back to the whole corpus rather than answering nothing; in comparison mode
    # that fallback would silently fill one regulation's slot with the other's
    # text, so callers there set strict_scope and an empty result stays empty.
    if plan.doc_kinds:
        scoped = [r for r in rows if r.get("doc_kind") in plan.doc_kinds]
        if scoped:
            rows = scoped
        elif strict_scope:
            return []

    bm25 = BM25Index([r["content"] for r in rows])
    keyword_scores = normalise(bm25.score(plan.question))

    results: list[RetrievedChunk] = []
    for index, row in enumerate(rows):
        semantic = cosine_similarity(query_embedding, row.get("embedding") or [])
        keyword = keyword_scores[index] if index < len(keyword_scores) else 0.0
        structure = _structure_score(row, plan)
        total = (
            settings.semantic_weight * semantic
            + settings.keyword_weight * keyword
            + settings.structure_weight * structure
            + _explicit_reference_bonus(row, plan)
        )
        results.append(
            RetrievedChunk(
                chunk_id=row["id"],
                document_id=row["document_id"],
                filename=row["filename"],
                short_label=row.get("short_label"),
                doc_kind=row.get("doc_kind", "general"),
                content=row["content"],
                page=row.get("page"),
                article=row.get("article"),
                article_num=row.get("article_num"),
                paragraph=row.get("paragraph"),
                section_path=row.get("section_path"),
                heading=row.get("heading"),
                chunk_kind=row.get("chunk_kind", "body"),
                score=total,
                semantic_score=semantic,
                keyword_score=keyword,
                structure_score=structure,
            )
        )

    results.sort(key=lambda c: c.score, reverse=True)
    return results[:top_k]


def retrieve_per_document_kind(
    conn: sqlite3.Connection,
    plan: QuestionPlan,
    query_embedding: list[float],
    doc_kinds: list[str],
    *,
    per_kind: int = 3,
) -> dict[str, list[RetrievedChunk]]:
    """Retrieve separately within each regulation, for comparison mode.

    Running one pooled query and splitting the results afterwards tends to
    return everything from whichever document phrases the topic more strongly;
    querying each side independently guarantees both are represented.
    """
    out: dict[str, list[RetrievedChunk]] = {}
    for kind in doc_kinds:
        scoped_plan = QuestionPlan(
            question=plan.question,
            question_type=plan.question_type,
            preferred_chunk_kinds=plan.preferred_chunk_kinds,
            doc_kinds=(kind,),
            article_refs=plan.article_refs,
            annex_refs=plan.annex_refs,
            is_comparison=plan.is_comparison,
            allows_direct_answer=False,
            quoted_terms=plan.quoted_terms,
        )
        out[kind] = retrieve(
            conn, scoped_plan, query_embedding, top_k=per_kind, strict_scope=True
        )
    return out


def compute_confidence(results: list[RetrievedChunk], plan: QuestionPlan) -> float:
    """Blend absolute strength, margin and structural agreement into 0..1.

    Absolute score alone is misleading — a weak corpus still produces a "best"
    chunk. The margin over the runner-up is what distinguishes "this clause
    answers it" from "several clauses are equally vague".
    """
    if not results:
        return 0.0

    top = results[0]
    # Semantic similarity is the honest signal of "did we find the topic".
    strength = max(0.0, min(top.semantic_score / 0.75, 1.0))

    if len(results) > 1 and top.score > 0:
        margin = (top.score - results[1].score) / top.score
    else:
        margin = 0.3
    margin = max(0.0, min(margin * 2.5, 1.0))

    agreement = 1.0 if (
        not plan.preferred_chunk_kinds or top.chunk_kind in plan.preferred_chunk_kinds
    ) else 0.55

    confidence = (0.55 * strength + 0.25 * margin + 0.20 * agreement)
    return round(max(0.0, min(confidence, 1.0)), 3)

"""Work out what a question is asking for before retrieving anything.

Classifying up front pays off twice: the retriever can bias toward chunks of a
matching kind (a "what is X" question should prefer `definition` chunks), and
the answer layer can decide whether the retrieved text can be returned directly
or needs the chat model to synthesise it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- Question type cues -------------------------------------------------------
DEFINITION_CUES = (
    r"\bwhat\s+(?:is|are)\b", r"\bdefine\b", r"\bdefinition\s+of\b",
    r"\bwhat\s+does\s+.+\s+mean\b", r"\bmeaning\s+of\b",
    # Role definitions are asked with "who", not "what": "Who is an authorised
    # representative?" is Article 2 lookup, not a question about duties.
    # "Who must…" / "Who is responsible…" stay obligations via OBLIGATION_CUES,
    # which is matched first.
    r"\bwho\s+(?:is|are)\s+(?:a|an|the)\b", r"\bwho\s+qualifies\s+as\b",
)
OBLIGATION_CUES = (
    r"\bobligation", r"\bwho\s+(?:must|shall|is\s+responsible)", r"\bresponsib",
    r"\brequired\s+to\b", r"\bduties\b", r"\bwhat\s+must\b", r"\bwhat\s+shall\b",
)
LIST_CUES = (
    r"\blist\b", r"\bwhat\s+are\s+the\b", r"\benumerate\b", r"\ball\s+the\b",
    r"\bwhich\s+.*\s+are\s+required\b",
)
CONDITION_CUES = (
    r"\bwhen\b", r"\bunder\s+what\s+(?:conditions|circumstances)\b", r"\bif\b",
    r"\bconditions?\s+for\b", r"\bexempt", r"\bderogat",
)
PROCEDURE_CUES = (
    r"\bhow\s+(?:to|do|does|can)\b", r"\bprocedure\b", r"\bprocess\b", r"\bsteps\b",
)
COMPARISON_CUES = (
    r"\bdifference[s]?\s+between\b", r"\bcompare\b", r"\bcompared\s+to\b",
    r"\bversus\b", r"\bvs\.?\b", r"\bboth\b", r"\bdiffer\b",
)
SUMMARY_CUES = (r"\bsummar", r"\boverview\b", r"\bin\s+short\b", r"\bexplain\b")

# --- Scope cues ---------------------------------------------------------------
RE_MDR = re.compile(r"\b(mdr|2017/745|medical\s+device\s+regulation)\b", re.IGNORECASE)
RE_IVDR = re.compile(
    r"\b(ivdr|2017/746|in\s*-?\s*vitro\s+diagnostic)\b", re.IGNORECASE
)
RE_ARTICLE_REF = re.compile(r"\barticle\s+(\d{1,3})\b", re.IGNORECASE)
RE_ANNEX_REF = re.compile(r"\bannex\s+([IVXLC]+|\d+)\b", re.IGNORECASE)

# Question types whose answer can be lifted straight from the source text.
DIRECT_ANSWERABLE = {"definition", "list"}


# "What is a 'medical device'?" / "Who is an authorised representative?" -> the term
RE_SUBJECT = re.compile(
    r"\b(?:what|who)\s+(?:is|are)\s+(?:a|an|the)?\s*['\"‘’“”]?(.+?)['\"‘’“”]?\s*[?.]?\s*$",
    re.IGNORECASE,
)


@dataclass
class QuestionPlan:
    question: str
    question_type: str = "general"
    preferred_chunk_kinds: tuple[str, ...] = ()
    doc_kinds: tuple[str, ...] = ()          # empty means "search everything"
    article_refs: list[int] = field(default_factory=list)
    annex_refs: list[str] = field(default_factory=list)
    is_comparison: bool = False
    allows_direct_answer: bool = False
    quoted_terms: list[str] = field(default_factory=list)
    # The term a definition question is about, used to match against the term a
    # definition block actually defines.
    subject_term: str | None = None


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _extract_quoted_terms(question: str) -> list[str]:
    """Pull out terms the user quoted or capitalised as a term of art."""
    terms = re.findall(r"['\"‘’“”](.+?)['\"‘’“”]", question)
    return [t.strip() for t in terms if 2 < len(t.strip()) < 80]


def plan_question(question: str) -> QuestionPlan:
    text = question.strip()
    plan = QuestionPlan(question=text)

    # --- type ---
    # Order is most-specific-first. "What are the obligations of manufacturers?"
    # matches the generic "what are" definition cue *and* the obligation cue, so
    # obligation has to be tested first or every such question becomes a lookup.
    if _matches_any(text, COMPARISON_CUES):
        plan.question_type = "comparison"
        plan.is_comparison = True
    elif _matches_any(text, OBLIGATION_CUES):
        plan.question_type = "obligation"
    elif _matches_any(text, DEFINITION_CUES):
        plan.question_type = "definition"
    elif _matches_any(text, LIST_CUES):
        plan.question_type = "list"
    elif _matches_any(text, CONDITION_CUES):
        plan.question_type = "condition"
    elif _matches_any(text, PROCEDURE_CUES):
        plan.question_type = "procedure"
    elif _matches_any(text, SUMMARY_CUES):
        plan.question_type = "summary"

    plan.preferred_chunk_kinds = {
        "definition": ("definition",),
        "obligation": ("obligation",),
        "list": ("list",),
        "condition": ("obligation", "list"),
        "procedure": ("obligation", "list"),
    }.get(plan.question_type, ())

    # --- scope ---
    wants_mdr = bool(RE_MDR.search(text))
    wants_ivdr = bool(RE_IVDR.search(text))
    if wants_mdr and wants_ivdr:
        plan.doc_kinds = ("mdr", "ivdr")
        plan.is_comparison = True
    elif wants_mdr:
        plan.doc_kinds = ("mdr",)
    elif wants_ivdr:
        plan.doc_kinds = ("ivdr",)

    plan.article_refs = [int(n) for n in RE_ARTICLE_REF.findall(text)]
    plan.annex_refs = [f"Annex {a.upper()}" for a in RE_ANNEX_REF.findall(text)]
    plan.quoted_terms = _extract_quoted_terms(text)

    if plan.question_type == "definition":
        if plan.quoted_terms:
            plan.subject_term = plan.quoted_terms[0].lower()
        elif match := RE_SUBJECT.search(text):
            candidate = match.group(1).strip().strip("'\"‘’“”").lower()
            if 2 < len(candidate) < 60:
                plan.subject_term = candidate

    # A comparison always needs synthesis, so it is never directly answerable.
    plan.allows_direct_answer = (
        plan.question_type in DIRECT_ANSWERABLE and not plan.is_comparison
    )
    return plan

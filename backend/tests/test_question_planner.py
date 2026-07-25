"""Tests for question classification and scoping."""

import pytest

from app.services.retrieval.question_planner import plan_question


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is a medical device?", "definition"),
        ("Define 'notified body'", "definition"),
        ("What are the obligations of manufacturers?", "obligation"),
        ("Who must draw up the declaration of conformity?", "obligation"),
        ("List the aspects of the quality management system", "list"),
        ("When does this Regulation not apply?", "condition"),
        ("How do I place a device on the market?", "procedure"),
        ("What is the difference between MDR and IVDR?", "comparison"),
        ("Summarise Chapter II", "summary"),
        # Role definitions are asked with "who". Regression: this was classified
        # "general" and took the slow synthesis path for an Article 2 lookup.
        ("Who is an 'authorised representative'?", "definition"),
        ("Who qualifies as a notified body?", "definition"),
        # …but "who must" is still about duties, not vocabulary.
        ("Who must draw up the technical documentation?", "obligation"),
        ("Who is responsible for post-market surveillance?", "obligation"),
    ],
)
def test_question_type_classification(question, expected):
    assert plan_question(question).question_type == expected


def test_mdr_scope_detected():
    plan = plan_question("What does the MDR say about importers?")
    assert plan.doc_kinds == ("mdr",)
    assert not plan.is_comparison


def test_ivdr_scope_detected():
    plan = plan_question("Under the IVDR, who is a manufacturer?")
    assert plan.doc_kinds == ("ivdr",)


def test_mentioning_both_regulations_triggers_comparison():
    plan = plan_question("How do MDR and IVDR treat manufacturers?")
    assert plan.is_comparison
    assert set(plan.doc_kinds) == {"mdr", "ivdr"}


def test_article_reference_is_extracted():
    plan = plan_question("What does Article 10 require?")
    assert plan.article_refs == [10]


def test_annex_reference_is_extracted():
    plan = plan_question("What is in Annex I?")
    assert plan.annex_refs == ["Annex I"]


def test_quoted_terms_are_extracted():
    plan = plan_question("What does 'notified body' mean?")
    assert "notified body" in plan.quoted_terms


def test_definitions_allow_direct_answers():
    assert plan_question("What is a medical device?").allows_direct_answer


def test_comparisons_never_allow_direct_answers():
    plan = plan_question("What is the difference between MDR and IVDR definitions?")
    assert not plan.allows_direct_answer


def test_obligation_questions_prefer_obligation_chunks():
    plan = plan_question("What must importers do?")
    assert "obligation" in plan.preferred_chunk_kinds


def test_unscoped_question_searches_everything():
    assert plan_question("What is a quality management system?").doc_kinds == ()

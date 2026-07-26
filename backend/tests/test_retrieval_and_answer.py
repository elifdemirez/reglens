"""Integration tests over an ingested two-regulation corpus."""

import pytest

from app.repositories import documents_repo
from app.services.document_processing.ingest import ingest_document
from app.services.rag.answer import answer_question, answer_question_stream
from app.services.rag.validate import NO_ANSWER_TEXT
from app.services.retrieval.hybrid import compute_confidence, retrieve
from app.services.retrieval.question_planner import plan_question
from tests.conftest import IVDR_SAMPLE, MDR_SAMPLE


@pytest.fixture
def corpus(conn, tmp_path, embedder):
    """Ingest both sample regulations and return their document ids."""
    ids = {}
    for label, text in (("mdr", MDR_SAMPLE), ("ivdr", IVDR_SAMPLE)):
        path = tmp_path / f"{label}.txt"
        path.write_text(text, encoding="utf-8")
        document_id = documents_repo.create(
            conn,
            filename=f"{label}.txt",
            stored_name=f"{label}.txt",
            file_type="txt",
            size_bytes=len(text),
        )
        ingest_document(conn, document_id, path, embedder)
        ids[label] = document_id
    return ids


# --- ingestion ----------------------------------------------------------------

def test_ingestion_marks_documents_ready(conn, corpus):
    rows = documents_repo.list_all(conn)
    assert len(rows) == 2
    assert all(row["status"] == "ready" for row in rows)
    assert all(row["chunk_count"] > 0 for row in rows)


def test_ingestion_labels_regulations(conn, corpus):
    labels = {row["doc_kind"]: row["short_label"] for row in documents_repo.list_all(conn)}
    assert labels["mdr"] == "MDR"
    assert labels["ivdr"] == "IVDR"


def test_stale_indexing_documents_are_recovered(conn):
    """Regression: an interrupted ingestion left the row stuck on 'indexing'
    forever, so the UI polled it indefinitely with no way to retry."""
    ready = documents_repo.create(
        conn, filename="done.txt", stored_name="a.txt", file_type="txt", size_bytes=10
    )
    documents_repo.mark_ready(
        conn, ready, chunk_count=3, page_count=1, doc_kind="mdr", short_label="MDR"
    )
    stuck = documents_repo.create(
        conn, filename="stuck.txt", stored_name="b.txt", file_type="txt", size_bytes=10
    )
    documents_repo.mark_indexing(conn, stuck)

    assert documents_repo.reset_stale_indexing(conn) == 1
    assert documents_repo.get(conn, stuck)["status"] == "failed"
    assert documents_repo.get(conn, stuck)["error"]
    # A completed document must not be touched.
    assert documents_repo.get(conn, ready)["status"] == "ready"


def test_ingestion_rejects_empty_file(conn, tmp_path, embedder):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    document_id = documents_repo.create(
        conn, filename="empty.txt", stored_name="empty.txt", file_type="txt", size_bytes=0
    )
    with pytest.raises(ValueError):
        ingest_document(conn, document_id, path, embedder)
    assert documents_repo.get(conn, document_id)["status"] == "failed"


# --- retrieval ----------------------------------------------------------------

def test_retrieval_finds_the_right_article(conn, corpus, embedder):
    plan = plan_question("What must manufacturers include in the quality management system?")
    results = retrieve(conn, plan, embedder.embed(plan.question))
    assert results
    assert any(r.article_num == 10 for r in results[:3])


def test_scoping_to_mdr_excludes_ivdr(conn, corpus, embedder):
    plan = plan_question("What does the MDR require of importers?")
    results = retrieve(conn, plan, embedder.embed(plan.question))
    assert results
    assert all(r.doc_kind == "mdr" for r in results)


def test_article_reference_boosts_that_article(conn, corpus, embedder):
    plan = plan_question("What does Article 13 say?")
    results = retrieve(conn, plan, embedder.embed(plan.question))
    assert results[0].article_num == 13


def test_citation_format_is_human_readable(conn, corpus, embedder):
    plan = plan_question("What does Article 10 require of manufacturers?")
    results = retrieve(conn, plan, embedder.embed(plan.question))
    citation = results[0].citation
    assert "Article 10" in citation
    assert results[0].short_label in citation


def test_citations_never_double_up_parentheses(conn, corpus, embedder):
    """Regression: definition chunks rendered as "IVDR, Article 2((1)), p. 13"."""
    plan = plan_question("What is a 'medical device'?")
    results = retrieve(conn, plan, embedder.embed(plan.question))
    assert results
    for chunk in results:
        assert "((" not in chunk.citation
        assert "))" not in chunk.citation


def test_confidence_is_higher_for_covered_questions(conn, corpus, embedder):
    covered = plan_question("What must manufacturers do about risk management?")
    unrelated = plan_question("What is the maximum tyre pressure for a tractor?")
    covered_conf = compute_confidence(
        retrieve(conn, covered, embedder.embed(covered.question)), covered
    )
    unrelated_conf = compute_confidence(
        retrieve(conn, unrelated, embedder.embed(unrelated.question)), unrelated
    )
    assert covered_conf > unrelated_conf


def test_retrieval_on_empty_corpus_returns_nothing(conn, embedder):
    plan = plan_question("Anything at all?")
    assert retrieve(conn, plan, embedder.embed(plan.question)) == []


# --- answering ----------------------------------------------------------------

def test_definition_question_takes_the_direct_path(conn, corpus, models):
    result = answer_question(conn, "What is a 'medical device'?", models)
    assert result.answer_path == "direct"
    assert "medical device" in result.answer.lower()
    # The direct path must not spend a chat-model call.
    assert models.call_count == 0


def test_definition_answer_skips_a_cross_reference(conn, corpus, models):
    """Regression: asking about 'medical device' returned IVDR Article 2(1), which
    only says the term means what MDR says it means — a dead end for the user.
    The substantive MDR definition must win instead."""
    result = answer_question(conn, "What is a 'medical device'?", models)
    assert result.answer_path == "direct"
    assert "as defined in point" not in result.answer
    assert "instrument" in result.answer.lower()
    assert "MDR" in result.answer


def test_definition_answer_matches_the_term_asked_about(conn, corpus, models):
    result = answer_question(conn, "What is an 'in vitro diagnostic medical device'?", models)
    assert "reagent" in result.answer.lower()


def test_obligation_question_uses_synthesis(conn, corpus, models):
    result = answer_question(conn, "What are the obligations of importers?", models)
    assert result.answer_path == "synthesis"
    assert models.call_count == 1


def test_context_passed_to_model_contains_citations(conn, corpus, models):
    answer_question(conn, "What are the obligations of importers?", models)
    assert "[" in models.last_user_prompt
    assert "Importers" in models.last_user_prompt


def test_unanswerable_question_is_refused(conn, corpus, models):
    result = answer_question(conn, "What is the tax rate on imported coffee in Brazil?", models)
    assert result.answer_path == "refused"
    assert result.answer == NO_ANSWER_TEXT


def test_model_self_refusal_is_reported_as_refused(conn, corpus, embedder):
    """Regression: when the model declined on its own, the result still claimed
    answer_path='synthesis', so the UI badge and history were misleading."""
    from tests.conftest import FakeChatModel

    declining = FakeChatModel(reply="The provided documents do not contain this information.")
    result = answer_question(conn, "What are the obligations of importers?", declining)
    assert result.answer_path == "refused"


def test_answer_reports_sources_with_scores(conn, corpus, models):
    result = answer_question(conn, "What are the obligations of importers?", models)
    assert result.sources
    assert all("citation" in s and "score" in s for s in result.sources)


def test_highlights_point_into_source_text(conn, corpus, models):
    # The canned reply paraphrases MDR Article 10(9), so this question is the
    # one where answer text and retrieved source genuinely overlap.
    result = answer_question(
        conn, "What must manufacturers include in the quality management system?", models
    )
    highlighted = [s for s in result.sources if s.get("highlights")]
    assert highlighted, "at least one source should have supporting spans"
    source = highlighted[0]
    for span in source["highlights"]:
        assert 0 <= span["start"] < span["end"] <= len(source["content"])


def test_comparison_mode_covers_both_regulations(conn, corpus, models):
    result = answer_question(
        conn, "What is the difference between MDR and IVDR manufacturer obligations?", models
    )
    assert result.mode == "compare"
    labels = {s["short_label"] for s in result.sources}
    assert {"MDR", "IVDR"} <= labels


def test_comparison_prompt_groups_by_regulation(conn, corpus, models):
    answer_question(conn, "Compare MDR and IVDR obligations for manufacturers", models)
    assert "=== MDR ===" in models.last_user_prompt
    assert "=== IVDR ===" in models.last_user_prompt


def test_comparison_refuses_when_only_one_regulation_indexed(conn, tmp_path, embedder, models):
    path = tmp_path / "mdr.txt"
    path.write_text(MDR_SAMPLE, encoding="utf-8")
    document_id = documents_repo.create(
        conn, filename="mdr.txt", stored_name="mdr.txt", file_type="txt", size_bytes=len(MDR_SAMPLE)
    )
    ingest_document(conn, document_id, path, embedder)

    result = answer_question(conn, "Compare MDR and IVDR manufacturer obligations", models)
    assert result.answer_path == "refused"
    assert "IVDR" in result.answer


def test_elapsed_time_is_recorded(conn, corpus, models):
    result = answer_question(conn, "What are the obligations of importers?", models)
    assert result.elapsed_ms >= 0


# --- streaming ----------------------------------------------------------------

def test_stream_emits_sources_before_tokens(conn, corpus, models):
    events = list(answer_question_stream(conn, "What are the obligations of importers?", models))
    types = [e["type"] for e in events]
    assert "sources" in types and "token" in types and types[-1] == "done"
    assert types.index("sources") < types.index("token")


def test_stream_final_event_matches_buffered_answer(conn, corpus, models):
    events = list(answer_question_stream(conn, "What are the obligations of importers?", models))
    done = events[-1]
    assert done["answer"].strip() == models.reply.strip()
    assert done["confidence"] > 0

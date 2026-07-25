"""End-to-end check against the real MDR/IVDR PDFs and real Foundry Local models.

Not part of the pytest suite: it downloads nothing but needs the sample PDFs in
``data/samples`` and takes several minutes on CPU. Run manually:

    python tests/e2e_real_models.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.repositories import documents_repo  # noqa: E402
from app.repositories.db import SCHEMA  # noqa: E402
from app.services.document_processing.ingest import ingest_document  # noqa: E402
from app.services.foundry.client import get_models  # noqa: E402
from app.services.rag.answer import answer_question  # noqa: E402

import sqlite3  # noqa: E402

SAMPLES = BACKEND_ROOT / "data" / "samples"
E2E_DB = BACKEND_ROOT / "data" / "e2e.db"
REPORT = BACKEND_ROOT / "data" / "e2e_report.json"

# (question, expectation) — expectations are deliberately about *grounding*
# (did it cite the right instrument / article), not about exact wording, which
# a 3.8B model will phrase differently each run.
QUESTIONS: list[tuple[str, dict]] = [
    ("What is a 'medical device'?", {"expect_kind": "mdr", "expect_type": "definition"}),
    ("What are the general obligations of importers?", {"expect_article": 13}),
    ("What must the quality management system cover?", {"expect_article": 10}),
    ("What does Article 10 require of manufacturers?", {"expect_article": 10}),
    ("Who is an 'authorised representative'?", {"expect_type": "definition"}),
    (
        "What is the difference between MDR and IVDR obligations for manufacturers?",
        {"expect_mode": "compare", "expect_both": True},
    ),
    (
        "What is the maximum permitted tyre pressure for agricultural tractors?",
        {"expect_refusal": True},
    ),
]


def build_corpus(conn: sqlite3.Connection, models) -> None:
    """Ingest each sample PDF, skipping any already indexed.

    Embedding ~1,150 chunks on CPU takes over 20 minutes, so this is resumable:
    a run interrupted partway through picks up where it left off instead of
    re-embedding what is already stored.
    """
    pdfs = sorted(SAMPLES.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {SAMPLES}")

    existing = {row["filename"]: row for row in documents_repo.list_all(conn)}

    for pdf in pdfs:
        previous = existing.get(pdf.name)
        if previous is not None and previous["status"] == "ready":
            print(
                f"\n--- skipping {pdf.name} (already indexed: "
                f"{previous['chunk_count']} chunks) ---",
                flush=True,
            )
            continue

        if previous is not None:
            # A stale 'indexing'/'failed' row from an interrupted run.
            documents_repo.delete(conn, previous["id"])

        print(f"\n--- ingesting {pdf.name} ({pdf.stat().st_size / 1e6:.1f} MB) ---", flush=True)
        started = time.perf_counter()
        document_id = documents_repo.create(
            conn,
            filename=pdf.name,
            stored_name=pdf.name,
            file_type="pdf",
            size_bytes=pdf.stat().st_size,
        )
        count = ingest_document(conn, document_id, pdf, models)
        row = documents_repo.get(conn, document_id)
        print(
            f"    {count} chunks, kind={row['doc_kind']}, label={row['short_label']}, "
            f"{time.perf_counter() - started:.0f}s",
            flush=True,
        )


def evaluate(conn: sqlite3.Connection, models) -> list[dict]:
    results = []
    for question, expectation in QUESTIONS:
        print(f"\n>>> {question}", flush=True)
        started = time.perf_counter()
        result = answer_question(conn, question, models)
        elapsed = time.perf_counter() - started

        articles = [s.get("article") for s in result.sources if s.get("article")]
        labels = {s.get("short_label") for s in result.sources}

        checks: dict[str, bool] = {}
        if "expect_article" in expectation:
            wanted = f"Article {expectation['expect_article']}"
            checks["cites_expected_article"] = wanted in articles
        if "expect_kind" in expectation:
            checks["cites_expected_regulation"] = any(
                s.get("doc_kind") == expectation["expect_kind"] for s in result.sources
            )
        if "expect_type" in expectation:
            checks["classified_correctly"] = result.question_type == expectation["expect_type"]
        if expectation.get("expect_mode"):
            checks["comparison_mode"] = result.mode == expectation["expect_mode"]
        if expectation.get("expect_both"):
            checks["both_regulations_cited"] = {"MDR", "IVDR"} <= labels
        if expectation.get("expect_refusal"):
            checks["refused_out_of_scope"] = result.answer_path == "refused"

        passed = all(checks.values()) if checks else True
        print(f"    path={result.answer_path} type={result.question_type} "
              f"conf={result.confidence:.2f} {elapsed:.1f}s", flush=True)
        print(f"    sources: {[s['citation'] for s in result.sources[:3]]}", flush=True)
        print(f"    answer : {result.answer[:200]!r}", flush=True)
        print(f"    checks : {checks} -> {'PASS' if passed else 'FAIL'}", flush=True)

        results.append(
            {
                "question": question,
                "answer": result.answer,
                "answer_path": result.answer_path,
                "question_type": result.question_type,
                "mode": result.mode,
                "confidence": result.confidence,
                "elapsed_sec": round(elapsed, 1),
                "citations": [s["citation"] for s in result.sources],
                "warnings": result.warnings,
                "checks": checks,
                "passed": passed,
            }
        )
    return results


def main() -> int:
    # The database is kept between runs so ingestion is resumable; pass --fresh
    # to force a full re-index.
    if "--fresh" in sys.argv:
        E2E_DB.unlink(missing_ok=True)
    conn = sqlite3.connect(E2E_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()

    print("Loading Foundry Local models…", flush=True)
    started = time.perf_counter()
    models = get_models()
    print(f"    ready in {time.perf_counter() - started:.0f}s", flush=True)

    build_corpus(conn, models)
    results = evaluate(conn, models)
    conn.close()

    passed = sum(1 for r in results if r["passed"])
    answer_times = [r["elapsed_sec"] for r in results]
    summary = {
        "passed": passed,
        "total": len(results),
        "avg_answer_sec": round(sum(answer_times) / len(answer_times), 1),
        "max_answer_sec": max(answer_times),
        "results": results,
    }
    REPORT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"E2E: {passed}/{len(results)} passed · "
          f"avg {summary['avg_answer_sec']}s · max {summary['max_answer_sec']}s")
    print(f"report -> {REPORT}")
    print("=" * 72)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

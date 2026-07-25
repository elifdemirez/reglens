import { useState } from "react";
import type { Source } from "../types";

/** Render the excerpt with the supporting spans wrapped in <mark>. */
function Excerpt({ source }: { source: Source }) {
  const highlights = (source.highlights ?? [])
    .slice()
    .sort((a, b) => a.start - b.start);

  if (highlights.length === 0) return <>{source.content}</>;

  const parts: React.ReactNode[] = [];
  let cursor = 0;
  highlights.forEach((span, index) => {
    // Spans are produced independently, so clamp and skip any that overlap
    // a previous one rather than emitting scrambled text.
    const start = Math.max(span.start, cursor);
    const end = Math.min(span.end, source.content.length);
    if (start >= end) return;
    if (start > cursor) parts.push(source.content.slice(cursor, start));
    parts.push(
      <mark key={index} title={`Supporting text (match ${(span.score * 100).toFixed(0)}%)`}>
        {source.content.slice(start, end)}
      </mark>,
    );
    cursor = end;
  });
  if (cursor < source.content.length) parts.push(source.content.slice(cursor));
  return <>{parts}</>;
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  return (
    <div>
      {label} {(value * 100).toFixed(0)}%
      <div className="bar">
        <i style={{ width: `${Math.min(Math.max(value, 0), 1) * 100}%` }} />
      </div>
    </div>
  );
}

export function SourceCard({ source, defaultOpen }: { source: Source; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(Boolean(defaultOpen));
  const label = source.short_label ?? source.filename;
  const tagClass = source.doc_kind === "mdr" ? "tag mdr" : source.doc_kind === "ivdr" ? "tag ivdr" : "tag";

  return (
    <div className="source">
      <button className="head" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className={tagClass}>{label}</span>
        <span className="citation">
          {source.article
            ? `${source.article}${source.paragraph ? `(${source.paragraph})` : ""}`
            : source.section_path ?? "Excerpt"}
          {source.page ? ` · p. ${source.page}` : ""}
        </span>
        <span className="score">{(source.score * 100).toFixed(0)}</span>
        <span aria-hidden>{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <>
          <div className="score-bars">
            <ScoreBar label="semantic" value={source.semantic_score} />
            <ScoreBar label="keyword" value={source.keyword_score} />
            <ScoreBar label="structure" value={source.structure_score} />
          </div>
          <div className="excerpt">
            <Excerpt source={source} />
          </div>
        </>
      )}
    </div>
  );
}

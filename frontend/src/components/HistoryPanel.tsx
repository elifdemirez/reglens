import { api } from "../api/client";
import type { HistoryEntry } from "../types";

export function HistoryPanel({
  entries,
  onChanged,
}: {
  entries: HistoryEntry[];
  onChanged: () => void;
}) {
  async function clear() {
    await api.clearHistory();
    onChanged();
  }

  return (
    <section className="panel">
      <header>
        <h2>Session history</h2>
        <span style={{ display: "flex", gap: 6 }}>
          <a href={api.exportHistoryUrl()} download="reglens-session.md">
            <button disabled={entries.length === 0}>Export</button>
          </a>
          <button className="ghost" disabled={entries.length === 0} onClick={() => void clear()}>
            Clear
          </button>
        </span>
      </header>
      <div className="body">
        {entries.length === 0 ? (
          <p className="hint">Questions you ask are recorded here and can be exported as Markdown.</p>
        ) : (
          entries.map((entry) => (
            <div className="history-item" key={entry.id}>
              <div className="q">{entry.question}</div>
              <div className="m">
                {(entry.confidence * 100).toFixed(0)}% · {entry.answer_path} ·{" "}
                {(entry.elapsed_ms / 1000).toFixed(1)}s
                {entry.sources[0] ? ` · ${entry.sources[0].citation}` : ""}
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

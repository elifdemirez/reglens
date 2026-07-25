import { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import { ChatPanel, type Turn } from "./components/ChatPanel";
import { DocumentPanel } from "./components/DocumentPanel";
import { HistoryPanel } from "./components/HistoryPanel";
import { SourceCard } from "./components/SourceCard";
import type { DocumentInfo, HealthInfo, HistoryEntry, Source } from "./types";

const POLL_MS = 2000;

export default function App() {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [docs, info, hist] = await Promise.all([
        api.listDocuments(),
        api.health(),
        api.history(),
      ]);
      setDocuments(docs);
      setHealth(info);
      setHistory(hist);
      setConnectionError(null);
    } catch (err) {
      setConnectionError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Ingestion runs in the background, so poll while anything is still indexing.
  useEffect(() => {
    const pending = documents.some((d) => d.status === "pending" || d.status === "indexing");
    if (!pending) return;
    const timer = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(timer);
  }, [documents, refresh]);

  const readyDocs = documents.filter((d) => d.status === "ready");
  const foundryOk = health?.foundry.available ?? false;

  const disabled = readyDocs.length === 0 || !foundryOk;
  const disabledReason = !foundryOk
    ? "Foundry Local is unavailable — start it and refresh."
    : "Upload a document first.";

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">
          <h1>RegLens</h1>
          <span>Local regulation analyst</span>
        </div>
        <div className="spacer" />
        <span className="status-pill">
          <i className={`dot ${foundryOk ? "ok" : "bad"}`} />
          Foundry Local {foundryOk ? "ready" : "offline"}
        </span>
        {health && (
          <span className="status-pill">
            <i className="dot" />
            {health.settings.chat_model}
          </span>
        )}
        <span className="status-pill">
          <i className={`dot ${readyDocs.length ? "ok" : ""}`} />
          {readyDocs.length} indexed
        </span>
      </div>

      {connectionError && (
        <div style={{ padding: "12px 24px 0" }}>
          <div className="error-box">
            Cannot reach the backend ({connectionError}). Start it with{" "}
            <code>uvicorn app.main:app --port 8000</code> in <code>backend/</code>.
          </div>
        </div>
      )}

      <div className="layout">
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <DocumentPanel
            documents={documents}
            limit={health?.documents.limit ?? 20}
            onChanged={() => void refresh()}
          />
          <HistoryPanel entries={history} onChanged={() => void refresh()} />
        </div>

        <ChatPanel
          turns={turns}
          setTurns={setTurns}
          onSources={setSources}
          onAnswered={() => void refresh()}
          disabled={disabled}
          disabledReason={disabledReason}
        />

        <section className="panel col-sources">
          <header>
            <h2>Sources</h2>
            <span className="count">{sources.length}</span>
          </header>
          <div className="body tight">
            {sources.length === 0 ? (
              <p className="hint" style={{ padding: 8 }}>
                Retrieved excerpts appear here, with the passages supporting the answer
                highlighted.
              </p>
            ) : (
              sources.map((source, index) => (
                <SourceCard key={source.chunk_id} source={source} defaultOpen={index === 0} />
              ))
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

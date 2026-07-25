import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { AnswerResult, Source, StreamEvent } from "../types";

export interface Turn {
  question: string;
  answer: string;
  result: AnswerResult | null;
  streaming: boolean;
  error?: string;
}

const SAMPLES = [
  "What is a 'medical device'?",
  "What are the obligations of importers?",
  "What must the quality management system cover?",
  "What is the difference between MDR and IVDR manufacturer obligations?",
];

function confidenceClass(value: number): string {
  if (value >= 0.6) return "badge high";
  if (value >= 0.35) return "badge medium";
  return "badge low";
}

function confidenceLabel(value: number): string {
  if (value >= 0.6) return "high";
  if (value >= 0.35) return "medium";
  return "low";
}

/** Render inline [Citation] markers as styled chips. */
function AnswerText({ text }: { text: string }) {
  const parts = text.split(/(\[[^\]]+\])/g);
  return (
    <>
      {parts.map((part, index) =>
        part.startsWith("[") && part.endsWith("]") ? (
          <span className="cite" key={index}>
            {part.slice(1, -1)}
          </span>
        ) : (
          <span key={index}>{part}</span>
        ),
      )}
    </>
  );
}

export function ChatPanel({
  turns,
  setTurns,
  onSources,
  onAnswered,
  disabled,
  disabledReason,
}: {
  turns: Turn[];
  setTurns: React.Dispatch<React.SetStateAction<Turn[]>>;
  onSources: (sources: Source[]) => void;
  /** Called once an answer completes, so the history panel picks it up. */
  onAnswered: () => void;
  disabled: boolean;
  disabledReason: string;
}) {
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState<string>("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, busy]);

  async function ask(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;

    setQuestion("");
    setBusy(true);
    setStage("planning");
    setTurns((prev) => [
      ...prev,
      { question: trimmed, answer: "", result: null, streaming: true },
    ]);

    const update = (patch: Partial<Turn>) =>
      setTurns((prev) => {
        const next = [...prev];
        next[next.length - 1] = { ...next[next.length - 1], ...patch };
        return next;
      });

    try {
      await api.askStream(trimmed, (event: StreamEvent) => {
        switch (event.type) {
          case "status":
            setStage(event.stage);
            break;
          case "sources":
            onSources(event.sources);
            break;
          case "token":
            setTurns((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              next[next.length - 1] = { ...last, answer: last.answer + event.text };
              return next;
            });
            break;
          case "done": {
            const { type: _type, replaced, ...result } = event;
            // The validator can swap a degenerate answer after it streamed,
            // so trust the final payload over the accumulated tokens.
            update({
              result: result as AnswerResult,
              answer: replaced ? result.answer : result.answer,
              streaming: false,
            });
            onSources(result.sources);
            break;
          }
          case "error":
            update({ error: event.message, streaming: false });
            break;
        }
      });
    } catch (err) {
      update({
        error: err instanceof Error ? err.message : String(err),
        streaming: false,
      });
    } finally {
      setBusy(false);
      setStage("");
      // The backend records every question; without this the history panel
      // would not show it until some other refresh happened to fire.
      onAnswered();
    }
  }

  return (
    <section className="panel col-chat">
      <header>
        <h2>Ask</h2>
        {busy && (
          <span className="thinking">
            {stage || "working"}
            <i />
            <i />
            <i />
          </span>
        )}
      </header>

      <div className="body">
        <div className="messages">
          {turns.length === 0 && (
            <div className="empty-state">
              <h3>Ask a question about your uploaded regulations</h3>
              Answers are generated locally and grounded in the document text.
              <div className="samples">
                {SAMPLES.map((sample) => (
                  <button key={sample} disabled={disabled} onClick={() => void ask(sample)}>
                    {sample}
                  </button>
                ))}
              </div>
            </div>
          )}

          {turns.map((turn, index) => (
            <div key={index}>
              <div className="msg user">{turn.question}</div>
              <div className="msg assistant" style={{ marginTop: 12 }}>
                {turn.error ? (
                  <div className="error-box">{turn.error}</div>
                ) : (
                  <>
                    <div className="answer">
                      <AnswerText text={turn.answer} />
                      {turn.streaming && !turn.answer && (
                        <span className="thinking">
                          <i />
                          <i />
                          <i />
                        </span>
                      )}
                    </div>
                    {turn.result && (
                      <>
                        <div className="msg-meta">
                          <span className={confidenceClass(turn.result.confidence)}>
                            confidence {confidenceLabel(turn.result.confidence)} ·{" "}
                            {(turn.result.confidence * 100).toFixed(0)}%
                          </span>
                          <span className="badge">{turn.result.question_type}</span>
                          <span className="badge">
                            {turn.result.answer_path === "direct"
                              ? "quoted from source"
                              : turn.result.answer_path === "refused"
                                ? "not in documents"
                                : "model synthesis"}
                          </span>
                          {turn.result.mode === "compare" && (
                            <span className="badge">comparison</span>
                          )}
                          <span className="badge">
                            {(turn.result.elapsed_ms / 1000).toFixed(1)}s
                          </span>
                        </div>
                        {turn.result.warnings.map((warning, i) => (
                          <div className="warning" key={i}>
                            {warning}
                          </div>
                        ))}
                      </>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="composer">
        <textarea
          value={question}
          placeholder={disabled ? disabledReason : "Ask about the uploaded regulations…"}
          disabled={disabled || busy}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void ask(question);
            }
          }}
        />
        <button
          className="primary"
          disabled={disabled || busy || !question.trim()}
          onClick={() => void ask(question)}
        >
          {busy ? "…" : "Ask"}
        </button>
      </div>
    </section>
  );
}

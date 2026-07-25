import { useRef, useState } from "react";
import { api } from "../api/client";
import type { DocumentInfo } from "../types";

const ACCEPT = ".txt,.md,.pdf,.docx";

function sizeLabel(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function statusLabel(doc: DocumentInfo): string {
  switch (doc.status) {
    case "pending":
      return "queued…";
    case "indexing":
      return "indexing…";
    case "failed":
      return "failed";
    default:
      return `${doc.chunk_count} chunks · ${doc.page_count} pages`;
  }
}

export function DocumentPanel({
  documents,
  limit,
  onChanged,
}: {
  documents: DocumentInfo[];
  limit: number;
  onChanged: () => void;
}) {
  const [dragOver, setDragOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function uploadFiles(files: FileList | File[]) {
    setError(null);
    setBusy(true);
    try {
      for (const file of Array.from(files)) {
        await api.uploadDocument(file);
        onChanged();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      onChanged();
    }
  }

  async function remove(id: number) {
    setError(null);
    try {
      await api.deleteDocument(id);
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <section className="panel col-docs">
      <header>
        <h2>Documents</h2>
        <span className="count">
          {documents.length}/{limit}
        </span>
      </header>
      <div className="body">
        <div
          className={dragOver ? "dropzone over" : "dropzone"}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            if (e.dataTransfer.files.length) void uploadFiles(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
        >
          <strong>{busy ? "Uploading…" : "Drop files or click"}</strong>
          PDF, DOCX, TXT, MD
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files?.length) void uploadFiles(e.target.files);
            e.target.value = "";
          }}
        />

        {error && (
          <div className="error-box" style={{ marginTop: 10 }}>
            {error}
          </div>
        )}

        <ul className="doc-list">
          {documents.map((doc) => (
            <li key={doc.id} className="doc">
              <div className="row">
                {doc.short_label && (
                  <span className={`tag ${doc.doc_kind}`}>{doc.short_label}</span>
                )}
                <span className="name" title={doc.filename}>
                  {doc.filename}
                </span>
                <button className="ghost" onClick={() => void remove(doc.id)} title="Delete">
                  ✕
                </button>
              </div>
              <div className="meta">
                {sizeLabel(doc.size_bytes)} · {statusLabel(doc)}
              </div>
              {doc.error && <div className="error">{doc.error}</div>}
            </li>
          ))}
        </ul>

        {documents.length === 0 && !busy && (
          <p className="hint" style={{ marginTop: 12 }}>
            Upload the MDR and IVDR to unlock comparison mode.
          </p>
        )}
      </div>
    </section>
  );
}

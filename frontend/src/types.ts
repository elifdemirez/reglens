export interface DocumentInfo {
  id: number;
  filename: string;
  file_type: string;
  size_bytes: number;
  status: "pending" | "indexing" | "ready" | "failed";
  doc_kind: string;
  short_label: string | null;
  chunk_count: number;
  page_count: number;
  error: string | null;
  created_at: string;
  indexed_at: string | null;
}

export interface Highlight {
  start: number;
  end: number;
  score: number;
}

export interface Source {
  chunk_id: number;
  document_id: number;
  filename: string;
  short_label: string | null;
  doc_kind: string;
  content: string;
  page: number | null;
  article: string | null;
  paragraph: string | null;
  section_path: string | null;
  heading: string | null;
  chunk_kind: string;
  citation: string;
  score: number;
  semantic_score: number;
  keyword_score: number;
  structure_score: number;
  highlights?: Highlight[];
}

export interface AnswerResult {
  answer: string;
  sources: Source[];
  confidence: number;
  answer_path: "direct" | "synthesis" | "refused";
  question_type: string;
  mode: "ask" | "compare";
  warnings: string[];
  elapsed_ms: number;
}

export interface HistoryEntry {
  id: number;
  question: string;
  answer: string;
  mode: string;
  answer_path: string;
  confidence: number;
  elapsed_ms: number;
  created_at: string;
  sources: { citation: string; score: number }[];
}

export interface HealthInfo {
  status: string;
  foundry: {
    available: boolean;
    models_loaded: boolean;
    chat_model: string;
    embedding_model: string;
    error?: string;
  };
  documents: { total: number; ready: number; limit: number };
  settings: {
    chat_model: string;
    embedding_model: string;
    top_k: number;
    max_file_size_mb: number;
  };
}

export type StreamEvent =
  | { type: "status"; stage: string; question_type?: string }
  | { type: "sources"; sources: Source[]; confidence: number }
  | { type: "token"; text: string }
  | ({ type: "done"; replaced?: boolean } & AnswerResult)
  | { type: "error"; message: string };

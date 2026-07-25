import type {
  AnswerResult,
  DocumentInfo,
  HealthInfo,
  HistoryEntry,
  StreamEvent,
} from "../types";

const BASE = "/api";

async function parseError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((d: { msg?: string }) => d.msg ?? "Invalid input").join("; ");
    }
    return response.statusText;
  } catch {
    return response.statusText || `Request failed (${response.status})`;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init);
  if (!response.ok) throw new Error(await parseError(response));
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthInfo>("/health"),

  listDocuments: () => request<DocumentInfo[]>("/documents"),

  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DocumentInfo>("/documents", { method: "POST", body: form });
  },

  deleteDocument: (id: number) =>
    request<void>(`/documents/${id}`, { method: "DELETE" }),

  ask: (question: string) =>
    request<AnswerResult>("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),

  history: () => request<HistoryEntry[]>("/history"),

  clearHistory: () => request<void>("/history", { method: "DELETE" }),

  exportHistoryUrl: () => `${BASE}/history/export`,

  /**
   * Ask with server-sent events so the answer appears while it is generated.
   * On a CPU-only machine a full answer can take ~30 s, so streaming is what
   * keeps the interface from looking frozen.
   */
  async askStream(
    question: string,
    onEvent: (event: StreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    const response = await fetch(`${BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, stream: true }),
      signal,
    });
    if (!response.ok) throw new Error(await parseError(response));
    if (!response.body) throw new Error("Streaming is not supported by this browser.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line; keep the trailing partial.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        try {
          onEvent(JSON.parse(line.slice(6)) as StreamEvent);
        } catch {
          // A malformed frame should not kill an otherwise working stream.
        }
      }
    }
  },
};

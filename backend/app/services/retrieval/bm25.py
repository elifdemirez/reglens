"""A small BM25 implementation (Okapi BM25, no external dependency).

Embeddings are good at "what does this mean", BM25 is good at "these exact
words". Legal text leans on fixed terms of art — *post-market surveillance*,
*notified body*, *conformity assessment* — where an embedding can drift to a
semantically adjacent but legally distinct clause. Running both and blending
the scores is what the hybrid retriever does.
"""

from __future__ import annotations

import math
import re
from collections import Counter

K1 = 1.5
B = 0.75

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately short: legal queries are keyword-poor to begin with, so pruning
# too aggressively removes signal.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were",
    "what", "which", "who", "with", "does", "do", "how", "when", "where", "why",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    def __init__(self, documents: list[str]) -> None:
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.doc_count = len(documents)
        self.avg_len = (sum(self.doc_len) / self.doc_count) if self.doc_count else 0.0

        self.term_freqs: list[Counter[str]] = [Counter(tokens) for tokens in self.doc_tokens]
        doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            doc_freq.update(set(tokens))

        self.idf: dict[str, float] = {}
        for term, freq in doc_freq.items():
            # Standard BM25 IDF with the +1 guard that keeps it non-negative for
            # terms appearing in more than half the corpus.
            self.idf[term] = math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))

    def score(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        scores = [0.0] * self.doc_count
        if not query_terms or not self.doc_count or self.avg_len == 0:
            return scores

        for index in range(self.doc_count):
            freqs = self.term_freqs[index]
            length = self.doc_len[index]
            total = 0.0
            for term in query_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denominator = tf + K1 * (1 - B + B * length / self.avg_len)
                total += idf * (tf * (K1 + 1)) / denominator
            scores[index] = total
        return scores


def normalise(scores: list[float]) -> list[float]:
    """Scale scores into 0..1 so they can be blended with cosine similarity."""
    if not scores:
        return []
    highest = max(scores)
    if highest <= 0:
        return [0.0] * len(scores)
    return [s / highest for s in scores]

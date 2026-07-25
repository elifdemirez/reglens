"""Shared fixtures.

The tests never load a real model. A hashing bag-of-words embedder stands in
for the embedding model: it is deterministic and, crucially, still puts texts
that share vocabulary close together, so retrieval assertions remain meaningful.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.repositories.db import SCHEMA  # noqa: E402
from app.services.retrieval.bm25 import tokenize  # noqa: E402

EMBED_DIM = 128


class FakeEmbedder:
    """Deterministic hashing embedder with cosine-comparable output."""

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * EMBED_DIM
        for token in tokenize(text):
            digest = hashlib.md5(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % EMBED_DIM
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class FakeChatModel(FakeEmbedder):
    """Embedder plus a scripted chat model."""

    def __init__(self, reply: str = "Manufacturers shall establish a quality management system. [MDR, Article 10(9)]"):
        self.reply = reply
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None
        self.call_count = 0

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.call_count += 1
        return self.reply

    def chat_stream(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None):
        self.chat(system_prompt, user_prompt, max_tokens)
        for word in self.reply.split(" "):
            yield word + " "


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def models() -> FakeChatModel:
    return FakeChatModel()


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    connection = sqlite3.connect(tmp_path / "test.db", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    connection.commit()
    yield connection
    connection.close()


# --- Sample corpora -----------------------------------------------------------
# Written to mirror EUR-Lex drafting conventions (chapters, numbered articles,
# numbered paragraphs, lettered points, definitions, annexes) without being a
# copy of any real instrument.

MDR_SAMPLE = """
Official Journal of the European Union
EN
L 117/1

REGULATION (EU) 2017/745 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL
on medical devices

CHAPTER I
SCOPE AND DEFINITIONS

Article 1
Subject matter and scope
1.   This Regulation lays down rules concerning the placing on the market of medical devices for human use.
2.   This Regulation does not apply to in vitro diagnostic medical devices.

Article 2
Definitions
For the purposes of this Regulation, the following definitions apply:
(1) 'medical device' means any instrument, apparatus, appliance, software or article intended by the manufacturer to be used for human beings for the diagnosis, prevention or treatment of disease.
(2) 'manufacturer' means a natural or legal person who manufactures a device and markets that device under its name.
(3) 'notified body' means a conformity assessment body designated in accordance with this Regulation.

CHAPTER II
OBLIGATIONS OF ECONOMIC OPERATORS

Article 10
General obligations of manufacturers
1.   Manufacturers shall ensure that their devices have been designed and manufactured in accordance with the requirements of this Regulation.
2.   Manufacturers shall establish, document and implement a risk management system.
9.   Manufacturers shall establish a quality management system that covers the following aspects:
(a) a strategy for regulatory compliance;
(b) identification of applicable general safety and performance requirements;
(c) resource management, including selection of suppliers;
(d) a post-market surveillance system.

Article 13
General obligations of importers
1.   Importers shall place on the market only devices that are in conformity with this Regulation.
2.   Importers shall verify that the device has been CE marked and that the declaration of conformity has been drawn up.

ANNEX I
GENERAL SAFETY AND PERFORMANCE REQUIREMENTS
Devices shall achieve the performance intended by the manufacturer and shall be designed so that they are suitable for their intended purpose.
"""

IVDR_SAMPLE = """
Official Journal of the European Union
EN
L 117/176

REGULATION (EU) 2017/746 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL
on in vitro diagnostic medical devices

CHAPTER I
SCOPE AND DEFINITIONS

Article 1
Subject matter and scope
1.   This Regulation lays down rules concerning the placing on the market of in vitro diagnostic medical devices for human use.

Article 2
Definitions
For the purposes of this Regulation, the following definitions apply:
(2) 'in vitro diagnostic medical device' means any medical device which is a reagent, calibrator, kit or system intended to be used in vitro for the examination of specimens derived from the human body.
(3) 'manufacturer' means a natural or legal person who manufactures a device and markets that device under its name.

CHAPTER II
OBLIGATIONS OF ECONOMIC OPERATORS

Article 10
General obligations of manufacturers
1.   Manufacturers shall ensure that their devices have been designed and manufactured in accordance with the requirements of this Regulation.
8.   Manufacturers shall establish a quality management system that covers the following aspects:
(a) a strategy for regulatory compliance;
(b) identification of applicable general safety and performance requirements;
(c) performance evaluation, including post-market performance follow-up.

Article 13
General obligations of importers
1.   Importers shall place on the market only devices that are in conformity with this Regulation.
"""


@pytest.fixture
def mdr_pages() -> list[tuple[int, str]]:
    return [(1, MDR_SAMPLE)]


@pytest.fixture
def ivdr_pages() -> list[tuple[int, str]]:
    return [(1, IVDR_SAMPLE)]

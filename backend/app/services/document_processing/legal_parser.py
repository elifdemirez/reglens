"""Structure-aware parsing for EUR-Lex style regulations.

EUR-Lex regulations (MDR, IVDR, and their siblings) follow a predictable
hierarchy::

    CHAPTER II
      SECTION 1
        Article 10
          1.  Manufacturers shall ...
              (a) ...
              (b) ...
    ANNEX I

Recovering that hierarchy is what makes a citation like *"MDR Article 10(4),
Chapter II"* possible, and it also gives the retrieval layer a structural
signal to boost on. Everything here is deliberately regex-based and tolerant:
PDF text extraction is noisy, so a block that fails to match simply stays
attached to whatever heading preceded it rather than being dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- Heading patterns ---------------------------------------------------------
# Roman numerals are matched loosely (IVXLC) because chapter numbering rarely
# goes past a couple of dozen in these instruments.
RE_CHAPTER = re.compile(r"^\s*CHAPTER\s+([IVXLC]+|\d+)\s*$", re.IGNORECASE)
RE_SECTION = re.compile(r"^\s*SECTION\s+(\d+|[IVXLC]+)\s*$", re.IGNORECASE)
RE_ANNEX = re.compile(r"^\s*ANNEX\s+([IVXLC]+|\d+)\s*$", re.IGNORECASE)
RE_ARTICLE = re.compile(r"^\s*Article\s+(\d+)\s*[a-z]?\s*$", re.IGNORECASE)
# Some extractions put the article title on the same line: "Article 10 Obligations"
RE_ARTICLE_INLINE = re.compile(r"^\s*Article\s+(\d+)\s*[a-z]?\s+(\S.*)$", re.IGNORECASE)

# --- Body patterns ------------------------------------------------------------
RE_PARAGRAPH = re.compile(r"^\s*(\d{1,2})\.\s+(\S.*)$")
# EUR-Lex PDFs frequently put the paragraph marker on a line of its own:
#     1.
#     When placing their devices on the market ...
# Without this the paragraph number never reaches the citation.
RE_PARAGRAPH_BARE = re.compile(r"^\s*(\d{1,2})\.\s*$")
RE_POINT = re.compile(r"^\s*\(([a-z]{1,2})\)\s+(\S.*)$")
RE_DEFINITION = re.compile(r"^\s*\((\d{1,3})\)\s+[‘'\"”“](.+?)[’'\"”“]\s+means\b", re.IGNORECASE)

# Modal verbs that signal a binding obligation in EU legal drafting.
RE_OBLIGATION = re.compile(r"\b(shall|must|are required to|is required to)\b", re.IGNORECASE)

# Page furniture that PDF extraction drags in and that only pollutes retrieval.
RE_NOISE = (
    re.compile(r"^\s*Official Journal of the European Union\s*$", re.IGNORECASE),
    re.compile(r"^\s*EN\s*$"),
    re.compile(r"^\s*L\s+\d+/\d+\s*$"),
    re.compile(r"^\s*\d{1,4}\s*$"),  # bare page numbers
    re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{4}\s*$"),  # dates in the header band
)


@dataclass
class Block:
    """A structurally-tagged span of text."""

    text: str
    page: int
    chapter: str | None = None
    section: str | None = None
    article: str | None = None
    article_num: int | None = None
    paragraph: str | None = None
    annex: str | None = None
    heading: str | None = None
    kind: str = "body"

    @property
    def section_path(self) -> str:
        parts = [p for p in (self.annex, self.chapter, self.section, self.article) if p]
        return " > ".join(parts)


@dataclass
class ParseResult:
    blocks: list[Block] = field(default_factory=list)
    detected_kind: str = "general"
    short_label: str | None = None
    article_count: int = 0


def _is_noise(line: str) -> bool:
    return any(pattern.match(line) for pattern in RE_NOISE)


# "(1) 'medical device' means any instrument, apparatus, ..." -> "medical device"
RE_DEFINED_TERM = re.compile(
    r"^\s*\(\d{1,3}\)\s*[‘'\"”“](.+?)[’'\"”“]\s+means\b", re.IGNORECASE
)

# Sister regulations define shared vocabulary by pointing at each other rather
# than repeating themselves, e.g. IVDR Article 2(1): "'medical device' means
# 'medical device' as defined in point (1) of Article 2 of Regulation (EU)
# 2017/745". Faithful, but a dead end as an answer.
RE_CROSS_REFERENCE = re.compile(
    r"\bas\s+defined\s+in\b.{0,120}?\b(Regulation|Directive|Article)\b",
    re.IGNORECASE | re.DOTALL,
)


def extract_defined_term(text: str) -> str | None:
    """Return the term a definition block defines, if it looks like one."""
    match = RE_DEFINED_TERM.match(text)
    return match.group(1).strip().lower() if match else None


def is_cross_reference_definition(text: str) -> bool:
    """True when a definition only redirects to another instrument's definition.

    Length matters: a substantive definition may *also* cite another article,
    but a pure redirect is short, so anything long enough to carry real content
    is treated as substantive.
    """
    stripped = text.strip()
    if len(stripped) > 320:
        return False
    return bool(RE_CROSS_REFERENCE.search(stripped))


# A regulation identifies itself in its title block. Matching that, rather than
# any mention of a number, is what keeps IVDR from being read as MDR: IVDR's
# definitions cite "Regulation (EU) 2017/745" repeatedly, so a bare number search
# misclassifies it the moment those pages fall inside the inspected window.
RE_SELF_TITLE = re.compile(
    r"REGULATION\s*\(EU\)\s*(\d{4}/\d+)\s+OF\s+THE\s+EUROPEAN\s+PARLIAMENT",
    re.IGNORECASE,
)

KNOWN_REGULATIONS = {
    "2017/745": ("mdr", "MDR"),
    "2017/746": ("ivdr", "IVDR"),
}


def detect_document_kind(text: str) -> tuple[str, str | None]:
    """Identify well-known regulations from their opening pages.

    Returns ``(kind, short_label)``. Unknown documents fall back to a generic
    "regulation" when they still look like structured legislation, otherwise
    "general".
    """
    head = text[:20000]

    if match := RE_SELF_TITLE.search(head):
        number = match.group(1)
        if number in KNOWN_REGULATIONS:
            return KNOWN_REGULATIONS[number]
        return "regulation", number

    # No title block (an extract, or a document that starts mid-way). Fall back to
    # whichever known regulation number appears first, since a document cites
    # others only after naming itself.
    positions = [
        (head.find(number), number)
        for number in KNOWN_REGULATIONS
        if head.find(number) != -1
    ]
    if positions:
        _, number = min(positions)
        return KNOWN_REGULATIONS[number]

    # Generic structured legislation: has articles and a regulation/directive title.
    if RE_ARTICLE.search(head) or re.search(r"^\s*Article\s+\d+", head, re.MULTILINE):
        match = re.search(
            r"(Regulation|Directive)\s*\(?E[UC]\)?\s*(\d{4}/\d+)", head, re.IGNORECASE
        )
        if match:
            return "regulation", match.group(2)
        return "regulation", None
    return "general", None


def _classify(text: str, *, in_annex: bool, is_definition: bool) -> str:
    if is_definition:
        return "definition"
    if in_annex:
        return "annex"
    # A block with several lettered points reads as an enumerated list, which the
    # answer layer can return verbatim instead of paraphrasing.
    if len(RE_POINT.findall(text)) >= 2 or text.count("\n(") >= 2:
        return "list"
    if RE_OBLIGATION.search(text):
        return "obligation"
    return "body"


def parse_pages(pages: list[tuple[int, str]]) -> ParseResult:
    """Walk the document line by line, tracking the current structural position."""
    full_text = "\n".join(text for _, text in pages)
    kind, label = detect_document_kind(full_text)

    blocks: list[Block] = []
    chapter: str | None = None
    section: str | None = None
    article: str | None = None
    article_num: int | None = None
    annex: str | None = None
    heading: str | None = None
    paragraph: str | None = None

    buffer: list[str] = []
    buffer_page = pages[0][0] if pages else 1
    buffer_is_definition = False
    articles_seen: set[int] = set()
    expecting_article_title = False

    def flush() -> None:
        nonlocal buffer, buffer_is_definition
        text = "\n".join(buffer).strip()
        buffer = []
        if not text:
            buffer_is_definition = False
            return
        blocks.append(
            Block(
                text=text,
                page=buffer_page,
                chapter=chapter,
                section=section,
                article=article,
                article_num=article_num,
                paragraph=paragraph,
                annex=annex,
                heading=heading,
                kind=_classify(text, in_annex=annex is not None, is_definition=buffer_is_definition),
            )
        )
        buffer_is_definition = False

    for page_number, page_text in pages:
        for raw_line in page_text.splitlines():
            line = raw_line.rstrip()
            if not line.strip() or _is_noise(line):
                continue

            # A heading closes whatever block was being accumulated.
            if match := RE_ANNEX.match(line):
                flush()
                annex = f"Annex {match.group(1).upper()}"
                chapter = section = article = None
                article_num = None
                paragraph = heading = None
                buffer_page = page_number
                continue

            if match := RE_CHAPTER.match(line):
                flush()
                chapter = f"Chapter {match.group(1).upper()}"
                section = None
                paragraph = None
                buffer_page = page_number
                continue

            if match := RE_SECTION.match(line):
                flush()
                section = f"Section {match.group(1)}"
                paragraph = None
                buffer_page = page_number
                continue

            if match := RE_ARTICLE.match(line):
                flush()
                article_num = int(match.group(1))
                article = f"Article {article_num}"
                articles_seen.add(article_num)
                annex = None
                paragraph = None
                heading = None
                expecting_article_title = True
                buffer_page = page_number
                continue

            if match := RE_ARTICLE_INLINE.match(line):
                flush()
                article_num = int(match.group(1))
                article = f"Article {article_num}"
                articles_seen.add(article_num)
                annex = None
                paragraph = None
                heading = match.group(2).strip()
                expecting_article_title = False
                buffer_page = page_number
                continue

            # The line right after a bare "Article N" is its title.
            if expecting_article_title:
                expecting_article_title = False
                if not RE_PARAGRAPH.match(line) and not RE_POINT.match(line) and len(line) < 200:
                    heading = line.strip()
                    continue

            if match := RE_DEFINITION.match(line):
                flush()
                # Stored bare, like every other paragraph marker; the citation
                # formatter adds the parentheses. Storing "(1)" here produced
                # citations such as "Article 2((1))".
                paragraph = match.group(1)
                buffer_page = page_number
                buffer.append(line.strip())
                buffer_is_definition = True
                continue

            if match := RE_PARAGRAPH.match(line):
                flush()
                paragraph = match.group(1)
                buffer_page = page_number
                buffer.append(line.strip())
                continue

            if match := RE_PARAGRAPH_BARE.match(line):
                flush()
                paragraph = match.group(1)
                buffer_page = page_number
                buffer.append(line.strip())
                continue

            buffer.append(line.strip())

    flush()

    if kind == "general" and len(articles_seen) >= 3:
        kind = "regulation"

    return ParseResult(
        blocks=blocks,
        detected_kind=kind,
        short_label=label,
        article_count=len(articles_seen),
    )

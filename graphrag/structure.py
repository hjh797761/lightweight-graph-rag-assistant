"""Conservative structure parsing with offsets into the unnormalized parsed text.

TXT/Markdown offsets address decoded file text, including original newlines.
DOCX offsets address paragraphs joined with ``\n``; PDF offsets address extracted
page texts joined with ``\n``. PDF text is not OCR and conveys no heading tree.
Numbered references support Chinese 第…章/节/条 and English chapter/section/clause
with Arabic dotted numbers. Bare list numbers and semantic references are ignored.
"""

from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
import re
import warnings

from .errors import DocumentParseError
from .models import EvidenceChunk, EvidenceLink, SourceBlock


_NUMBER = r"[0-9一二三四五六七八九十百零〇两]+"
_REFERENCE = re.compile(
    rf"第\s*(?P<zh>{_NUMBER})\s*(?P<unit>[章节条])"
    r"|\b(?P<en>chapter|section|clause)\s*(?P<number>\d+(?:\.\d+)*)(?!\w|\.[\w.])",
    re.IGNORECASE,
)
_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?)|[ \t]*)$")


def _trim(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _headings(text, format):
    """Return explicit heading offsets/levels/titles, excluding fenced examples."""
    result = []
    offset = 0
    fence = None
    lines = text.splitlines(keepends=True)
    underline_index = -1
    for index, line in enumerate(lines):
        value = line.rstrip("\r\n")
        if format == "markdown":
            marker = re.match(r"^ {0,3}(`{3,}|~{3,})", value)
            if marker:
                run = marker.group(1)
                if fence is None:
                    fence = run
                elif run[0] == fence[0] and len(run) >= len(fence) and not value[marker.end():].strip():
                    fence = None
            elif fence is None and index != underline_index:
                match = _ATX.match(value)
                if match:
                    title = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2) or "").strip()
                    result.append((offset, len(match.group(1)), title, offset + len(value)))
                elif value.strip() and not re.match(r"^(?: {4}|\t|\s*[-*+>]\s)", value) and index + 1 < len(lines):
                    underline = re.fullmatch(r" {0,3}(=+|-+)[ \t]*", lines[index + 1].rstrip("\r\n"))
                    if underline:
                        result.append((offset, 1 if underline.group(1)[0] == "=" else 2, value.strip(),
                                       offset + len(line) + len(lines[index + 1].rstrip("\r\n"))))
                        underline_index = index + 1
        else:
            match = _REFERENCE.match(value.lstrip())
            if match and (match.end() == len(value.lstrip()) or value.lstrip()[match.end()].isspace()):
                unit = match.group("unit") or match.group("en").lower()
                level = {"章": 1, "chapter": 1, "节": 2, "section": 2, "条": 3, "clause": 3}[unit]
                result.append((offset, level, value.strip(), offset + len(value)))
        offset += len(line)
    return result


def _blocks(text, doc_id, headings):
    boundaries = list(headings)
    if not boundaries or boundaries[0][0] > 0:
        boundaries.insert(0, (0, 0, None, None))
    stack = []
    result = []
    for index, (start, level, title, heading_end) in enumerate(boundaries):
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(text)
        if title is not None:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        section_id = f"{doc_id}::section_{start}" if title is not None else None
        result.append(SourceBlock(text[start:end], section_id, tuple(t for _, t in stack),
                                  start=start, end=end, heading_end=heading_end))
    return result


def _spans(text, start, end, chunk_size):
    """Prefer paragraphs then sentences; only oversized sentences become partial."""
    start, end = _trim(text, start, end)
    if start == end:
        return []
    if end - start <= chunk_size:
        return [(start, end, False)]
    parts = []
    cursor = start
    # A newline is a safe paragraph/heading boundary, preserving raw coordinates.
    for match in re.finditer(r"\r?\n+", text[start:end]):
        edge = start + match.start()
        if cursor < edge:
            parts.append((cursor, edge))
        cursor = start + match.end()
    if cursor < end:
        parts.append((cursor, end))
    atoms = []
    for a, b in parts:
        a, b = _trim(text, a, b)
        if b - a <= chunk_size:
            if a < b:
                atoms.append((a, b, False))
            continue
        cursor = a
        for match in re.finditer(r"[。！？!?；;]+[\"'”’）)]*|\.(?=\s|$)", text[a:b]):
            edge = a + match.end()
            atoms.extend(_sentence_spans(text, cursor, edge, chunk_size))
            cursor = edge
        atoms.extend(_sentence_spans(text, cursor, b, chunk_size))
    merged = []
    for a, b, partial in atoms:
        if merged and not partial and not merged[-1][2] and b - merged[-1][0] <= chunk_size:
            merged[-1] = (merged[-1][0], b, False)
        else:
            merged.append((a, b, partial))
    return merged


def _sentence_spans(text, start, end, chunk_size):
    start, end = _trim(text, start, end)
    partial = end - start > chunk_size
    return [(a, min(a + chunk_size, end), partial) for a in range(start, end, chunk_size)]


def _chunks(text, doc_id, blocks, chunk_size, locator):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    result = []
    for block in blocks:
        previous_end = None
        for start, end, partial in _spans(text, block.start, block.end, chunk_size):
            raw = text[start:end]
            result.append(EvidenceChunk(
                id=f"{doc_id}::chunk_{len(result):06d}", doc_id=doc_id,
                sequence=len(result), clean_text=raw, context_text=raw,
                chapter=block.title_path[-1] if block.title_path else "未分类",
                section_id=block.section_id, title_path=block.title_path,
                locator=locator(start, end), source_start=start, source_end=end,
                partial=partial, source_text=raw, heading_end=block.heading_end,
                source_gap_before=text[previous_end:start] if previous_end is not None else "",
            ))
            previous_end = end
    return result


def _line_locator(text, label="lines"):
    starts = [0] + [m.end() for m in re.finditer(r"\n", text)]
    return lambda start, end: f"{label} {bisect_right(starts, start)}-{bisect_right(starts, max(start, end - 1))}"


def parse_text(text: str, doc_id: str, format: str = "markdown", chunk_size: int = 800) -> list[EvidenceChunk]:
    format = {"md": "markdown", "text": "txt"}.get(format.lower(), format.lower())
    if format not in {"markdown", "txt"}:
        raise ValueError(f"unsupported format: {format}")
    return _chunks(text, doc_id, _blocks(text, doc_id, _headings(text, format)), chunk_size, _line_locator(text))


def parse_file(path: str | Path, doc_id: str, chunk_size: int = 800) -> list[EvidenceChunk]:
    path = Path(path)
    suffix = path.suffix.lower()
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    try:
        if suffix in {".md", ".txt"}:
            with path.open(encoding="utf-8", newline="") as source:
                return parse_text(source.read(), doc_id, suffix[1:], chunk_size)
        if suffix == ".docx":
            from docx import Document

            paragraphs = Document(path).paragraphs
            text = "\n".join(p.text for p in paragraphs)
            headings, starts = [], []
            offset = 0
            for paragraph in paragraphs:
                starts.append(offset)
                style = paragraph.style
                name = style.name if style is not None else ""
                match = re.fullmatch(r"Heading\s+([1-9])", name, re.IGNORECASE)
                if match:
                    headings.append((offset, int(match.group(1)), paragraph.text.strip(), offset + len(paragraph.text)))
                offset += len(paragraph.text) + 1
            locator = lambda a, b: f"paragraphs {bisect_right(starts, a)}-{bisect_right(starts, max(a, b - 1))}"
            return _chunks(text, doc_id, _blocks(text, doc_id, headings), chunk_size, locator)
        if suffix == ".pdf":
            # Keep the evidence path free of the native MuPDF/torch shutdown
            # interaction reproduced in this Windows CPU environment. PyPDF2
            # is already a required dependency; it needs no newer MuPDF import.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"PyPDF2 is deprecated\. Please move to the pypdf library instead\.",
                                        category=DeprecationWarning, module=r"PyPDF2")
                from PyPDF2 import PdfReader
            with path.open("rb") as stream:
                pages = [page.extract_text() or "" for page in PdfReader(stream).pages]
            text = "\n".join(pages)
            blocks, starts = [], []
            offset = 0
            for number, page in enumerate(pages, 1):
                starts.append(offset)
                blocks.append(SourceBlock(page, locator=f"page {number}", start=offset, end=offset + len(page)))
                offset += len(page) + 1
            locator = lambda a, b: f"page {bisect_right(starts, a)}"
            return _chunks(text, doc_id, blocks, chunk_size, locator)
    except Exception as exc:
        raise DocumentParseError(f"无法读取文档 {path}") from exc
    raise DocumentParseError(f"不支持的文档类型: {suffix or '无扩展名'}")


def _label(match):
    if match.group("zh"):
        return match.group("unit"), match.group("zh")
    return match.group("en").lower(), match.group("number")


def _reference_spans(members):
    """Recover references from exact slices and parser-recorded original gaps."""
    runs = []
    for chunk in members:
        if (runs and runs[-1][-1].doc_id == chunk.doc_id
                and runs[-1][-1].source_end + len(chunk.source_gap_before) == chunk.source_start
                and runs[-1][-1].section_id == chunk.section_id):
            runs[-1].append(chunk)
        else:
            runs.append([chunk])
    for run in runs:
        source = "".join((chunk.source_gap_before if index else "") + chunk.source_text
                         for index, chunk in enumerate(run))
        starts = [chunk.source_start for chunk in run]
        for match in _REFERENCE.finditer(source):
            start = starts[0] + match.start()
            chunk = run[bisect_right(starts, start) - 1]
            yield chunk, match, start, starts[0] + match.end()


def build_links(chunks: list[EvidenceChunk]) -> list[EvidenceLink]:
    """Build only placement, direct same-section adjacency, and literal references.

    Section membership targets a section ID; resolved references target the first
    chunk and retain every target chunk in target_ids. Adjacency is bidirectional.
    """
    groups = defaultdict(list)
    documents = defaultdict(list)
    for chunk in chunks:
        documents[chunk.doc_id].append(chunk)
        if chunk.section_id is not None:
            groups[(chunk.doc_id, chunk.section_id)].append(chunk)
    labels = defaultdict(list)
    for (doc_id, section_id), members in groups.items():
        members.sort(key=lambda c: c.sequence)
        first = members[0]
        title = first.title_path[-1] if first.title_path else ""
        match = _REFERENCE.match(title)
        if match:
            labels[(doc_id, _label(match))].append(members)
    links = []
    for doc_id, members in documents.items():
        members.sort(key=lambda c: c.sequence)
        for index, chunk in enumerate(members):
            if chunk.section_id is not None:
                links.append(EvidenceLink(chunk.id, chunk.section_id, "section_member", chunk.locator,
                                          source_start=chunk.source_start, source_end=chunk.source_end))
                if index and members[index - 1].section_id == chunk.section_id and members[index - 1].sequence + 1 == chunk.sequence:
                    previous = members[index - 1]
                    links.extend((EvidenceLink(previous.id, chunk.id, "adjacent", "consecutive chunks in same section"),
                                  EvidenceLink(chunk.id, previous.id, "adjacent", "consecutive chunks in same section")))
        for chunk, match, start, end in _reference_spans(members):
            if chunk.heading_end is not None and start < chunk.heading_end:
                continue
            candidates = labels.get((doc_id, _label(match)), [])
            status = "resolved" if len(candidates) == 1 else "missing" if not candidates else "ambiguous"
            targets = tuple(c.id for c in candidates[0]) if status == "resolved" else ()
            links.append(EvidenceLink(chunk.id, targets[0] if targets else None,
                                      "explicit_reference", match.group(), status, start, end, targets))
    return links

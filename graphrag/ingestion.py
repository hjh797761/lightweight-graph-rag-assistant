from pathlib import Path
import hashlib
import re

import numpy as np

from .errors import DocumentParseError
from .graph import extract_concepts
from .models import ChunkRecord


def make_doc_id(file_path: str | Path) -> str:
    path = Path(file_path).resolve()
    safe_name = re.sub(r"\W+", "_", path.stem, flags=re.UNICODE).strip("_")[:30]
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:12]
    return f"{safe_name or 'document'}_{digest}"


def load_text(file_path: str | Path) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8")
        if suffix == ".docx":
            from docx import Document

            return "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
        if suffix == ".pdf":
            try:
                import fitz

                document = fitz.open(path)
                return "\n".join(
                    f"[[PAGE:{index + 1}]]\n{page.get_text()}"
                    for index, page in enumerate(document)
                )
            except ImportError:
                from PyPDF2 import PdfReader

                return "\n".join(
                    f"[[PAGE:{index + 1}]]\n{page.extract_text() or ''}"
                    for index, page in enumerate(PdfReader(path).pages)
                )
    except Exception as exc:
        raise DocumentParseError(f"无法读取文档 {path}") from exc
    raise DocumentParseError(f"不支持的文档类型: {suffix or '无扩展名'}")


def split_text(text: str, chunk_size: int = 800) -> list[tuple[str, str]]:
    text = (text or "").strip()
    if not text:
        return []
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n+", text) if item.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
        else:
            chunks.extend(
                paragraph[index : index + chunk_size].strip()
                for index in range(0, len(paragraph), chunk_size)
                if paragraph[index : index + chunk_size].strip()
            )
    overlap = min(120, max(40, chunk_size // 8))
    return [
        (chunk, f"{chunks[index - 1][-overlap:]}\n{chunk}" if index else chunk)
        for index, chunk in enumerate(chunks)
    ]


def _chapter(text: str) -> str:
    first_line = (text.splitlines() or [""])[0].strip(" #：:")
    if 1 < len(first_line) <= 30:
        return first_line
    return "未分类"


def ingest_text(
    doc_id: str,
    text: str,
    store,
    embedder,
    batch_size: int,
    *,
    doc_name: str = "",
    doc_path: str = "",
) -> tuple[int, int]:
    pairs = split_text(text)
    start = store.get_progress(doc_id)
    selected = pairs[start : start + batch_size]
    if not selected:
        return start, len(pairs)
    vectors = embedder.encode_many(
        [clean for clean, _ in selected], batch_size=batch_size
    )
    if len(vectors) != len(selected):
        raise ValueError("嵌入数量与片段数量不一致")
    chunks = [
        ChunkRecord(
            id=f"{doc_id}::chunk_{start + offset:06d}",
            doc_id=doc_id,
            sequence=start + offset,
            clean_text=clean,
            context_text=context,
            chapter=_chapter(clean),
            concepts=extract_concepts(clean),
            embedding=np.asarray(vectors[offset], dtype=np.float32),
        )
        for offset, (clean, context) in enumerate(selected)
    ]
    next_index = start + len(chunks)
    store.write_chunk_batch(
        doc_id,
        chunks,
        next_index=next_index,
        doc_name=doc_name,
        doc_path=doc_path,
    )
    return next_index, len(pairs)


def ingest_file(file_path: str | Path, store, embedder, batch_size: int) -> tuple[int, int]:
    path = Path(file_path).resolve()
    return ingest_text(
        make_doc_id(path),
        load_text(path),
        store,
        embedder,
        batch_size,
        doc_name=path.name,
        doc_path=str(path),
    )

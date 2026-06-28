"""Step 3: chunking with metadata.

Strategy (per ingestion_pipeline.md):
  * Respect structural boundaries first - never split mid-Block unless it
    exceeds CHUNK_MAX_TOKENS.
  * Target ~CHUNK_TARGET_TOKENS tokens per chunk (cl100k via tiktoken).
  * Hard maximum at CHUNK_MAX_TOKENS.
  * No overlap by default - retrieval can grab adjacent chunks via
    section-reference metadata.

Each chunk JSON record contains structural metadata (jurisdiction, issuing
body, document id/hash, section/article reference, document type, retrieval
date, source URL placeholder) and a topical-tag stub left for Step 3b.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

from . import config
from .parse.base import Block

_ENC = tiktoken.get_encoding("cl100k_base")


def _ntok(s: str) -> int:
    return len(_ENC.encode(s, disallowed_special=()))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    jurisdiction: str
    issuing_body: str
    document_type: str
    filename: str
    source_path: str
    document_hash: str
    citation: str
    parent_citation: str | None
    kind: str
    page_start: int | None
    page_end: int | None
    retrieval_date: str
    effective_date: str | None
    source_url: str | None
    text: str
    n_tokens: int
    regime_tags: list[str] = field(default_factory=list)
    topic_tags: list[str] = field(default_factory=list)
    applicability_tags: list[str] = field(default_factory=list)


def _split_long_block(block: Block, max_tokens: int, target_tokens: int) -> list[str]:
    """Split a block's text along sentence boundaries to keep each chunk <= max_tokens
    while staying near target_tokens.
    """
    text = block.text
    if _ntok(text) <= max_tokens:
        return [text]

    # Split on paragraph blanks first, then on sentence endings.
    paragraphs = re.split(r"\n{2,}", text)
    sentences: list[str] = []
    for para in paragraphs:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", para.strip())
        sentences.extend(p for p in parts if p)

    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for sent in sentences:
        st = _ntok(sent)
        if st > max_tokens:
            # Pathological long sentence - hard split on tokens directly.
            if buf:
                chunks.append(" ".join(buf))
                buf, buf_tokens = [], 0
            ids = _ENC.encode(sent, disallowed_special=())
            for i in range(0, len(ids), target_tokens):
                chunks.append(_ENC.decode(ids[i:i + target_tokens]))
            continue

        if buf_tokens + st > max_tokens:
            chunks.append(" ".join(buf))
            buf, buf_tokens = [sent], st
            continue

        buf.append(sent)
        buf_tokens += st
        if buf_tokens >= target_tokens:
            chunks.append(" ".join(buf))
            buf, buf_tokens = [], 0

    if buf:
        chunks.append(" ".join(buf))
    return chunks


def chunk_blocks(
    blocks: list[Block],
    *,
    doc_meta: dict,
    target_tokens: int = config.CHUNK_TARGET_TOKENS,
    max_tokens: int = config.CHUNK_MAX_TOKENS,
) -> list[Chunk]:
    """Convert a parsed document into Chunk records.

    Adjacent small blocks within the same parent citation are merged up to
    target_tokens to avoid an explosion of tiny chunks (common in MAS Notices
    where subclauses 3.2.1, 3.2.2 each have 2-3 sentences).
    """
    chunks: list[Chunk] = []
    today = _dt.date.today().isoformat()

    buffer_texts: list[str] = []
    buffer_block: Block | None = None
    buffer_tokens = 0

    def flush():
        nonlocal buffer_texts, buffer_block, buffer_tokens
        if not buffer_texts or buffer_block is None:
            buffer_texts, buffer_tokens = [], 0
            return
        text = "\n\n".join(buffer_texts).strip()
        if text:
            chunks.append(_make_chunk(buffer_block, text, doc_meta, today))
        buffer_texts, buffer_block, buffer_tokens = [], None, 0

    for blk in blocks:
        blk_tokens = _ntok(blk.text)

        # Big block: flush buffer, then split this block individually.
        if blk_tokens > max_tokens:
            flush()
            for piece in _split_long_block(blk, max_tokens, target_tokens):
                if piece.strip():
                    chunks.append(_make_chunk(blk, piece, doc_meta, today))
            continue

        # Can we accumulate this block onto the buffer?
        # Only if same parent citation (or both top-level) and we'd stay <= target.
        same_parent = (
            buffer_block is not None
            and buffer_block.parent_citation == blk.parent_citation
            and buffer_block.kind == blk.kind
        )
        if same_parent and (buffer_tokens + blk_tokens) <= target_tokens:
            buffer_texts.append(blk.text)
            buffer_tokens += blk_tokens
            continue

        # Otherwise: flush and start a new buffer with this block.
        flush()
        buffer_texts = [blk.text]
        buffer_block = blk
        buffer_tokens = blk_tokens

    flush()
    return chunks


def _make_chunk(blk: Block, text: str, doc_meta: dict, today: str) -> Chunk:
    n = _ntok(text)
    cid = f"{doc_meta['doc_id']}::{len(text):x}::{hash(text) & 0xffff:04x}"
    return Chunk(
        chunk_id=cid,
        doc_id=doc_meta["doc_id"],
        jurisdiction=doc_meta["jurisdiction"],
        issuing_body=doc_meta["issuing_body"],
        document_type=doc_meta.get("document_type", ""),
        filename=doc_meta["filename"],
        source_path=doc_meta["source_path"],
        document_hash=doc_meta["doc_id"],  # full file hash truncated, same as doc_id
        citation=blk.citation,
        parent_citation=blk.parent_citation,
        kind=blk.kind,
        page_start=blk.page_start,
        page_end=blk.page_end,
        retrieval_date=today,
        effective_date=doc_meta.get("effective_date"),
        source_url=doc_meta.get("source_url"),
        text=text,
        n_tokens=n,
    )


def write_chunks_jsonl(chunks: list[Chunk], doc_meta: dict) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(doc_meta["filename"]).stem)
    out = (
        config.CHUNKS_DIR
        / doc_meta["jurisdiction"]
        / doc_meta["issuing_body"]
        / f"{safe}__{doc_meta['doc_id']}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    return out

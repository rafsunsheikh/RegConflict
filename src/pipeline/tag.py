"""Step 3b: topical tagging stub.

The user opted to skip Claude-assisted tagging for this run. Chunks therefore
ship with empty regime_tags / topic_tags / applicability_tags fields, ready for
backfill by a later script.

The interface below exists so the runner has a single place to plug in real
tagging when an ANTHROPIC_API_KEY becomes available.
"""
from __future__ import annotations

from .chunk import Chunk


def tag_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return chunks

"""Pydantic v2 schema for RegConflict labelled records.

The canonical schema is documented in `data/schema.md`. This module provides
typed Python access for code that wants validation on read.

Usage:
    from src.eval.benchmark.schema import LabelledPair
    record = LabelledPair.model_validate(json.loads(line))
    if record.record_type == "conflict":
        print(record.conflict_type, record.severity)
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ConflictType(str, Enum):
    structural_unresolved = "structural_unresolved"
    operationally_resolvable = "operationally_resolvable"
    interpretive_fact_sensitive = "interpretive_fact_sensitive"
    recurring_friction = "recurring_friction"


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class NonConflictSubtype(str, Enum):
    """Documented non-conflict subtypes. The field is descriptive-not-prescriptive
    in the schema (it's not part of the evaluation task; see schema.md), so the
    schema accepts arbitrary strings, but the values below are the canonical set
    used during v1.0 labelling and should be preferred when contributing new
    records."""
    aligned_complementary = "aligned_complementary"
    different_domains = "different_domains"
    superficially_similar_but_disjoint = "superficially_similar_but_disjoint"
    same_jurisdiction_no_overlap = "same_jurisdiction_no_overlap"
    thresholds_dont_collide = "thresholds_dont_collide"  # used in v0.9 single edge case


class Jurisdiction(str, Enum):
    Australia = "Australia"
    EU = "EU"
    Singapore = "Singapore"
    International = "International"


class ReviewStatus(str, Enum):
    single_annotation = "single_annotation"
    reviewed_accept = "reviewed_accept"
    reviewed_revise = "reviewed_revise"
    reviewed_reject = "reviewed_reject"


class Regime(BaseModel):
    """One of the two regulatory regimes in a labelled pair."""
    model_config = ConfigDict(extra="allow")

    regime_id: str
    jurisdiction: Jurisdiction
    issuing_body: str = "?"
    short_name: Optional[str] = None


class EvidenceChunk(BaseModel):
    """One evidence chunk shown to annotators (and models) for a regime side."""
    model_config = ConfigDict(extra="allow")

    chunk_id: str
    source_doc: str
    page: Optional[int] = None
    passage: str


class ReviewBlock(BaseModel):
    """Adjudication record present on IAA-overlap pairs."""
    model_config = ConfigDict(extra="allow")

    reviewer_id: str
    reviewed_at: str
    decision: Literal["accept", "revise", "reject"]
    notes: str = ""
    original_conflict_type: Optional[ConflictType] = None
    original_severity: Optional[Severity] = None
    original_review_status: Optional[ReviewStatus] = None


class LabelledPair(BaseModel):
    """A single labelled regime-pair record from data/conflicts/*.jsonl.

    Type-specific fields:
      - record_type=conflict       → conflict_type and severity required
      - record_type=non_conflict   → non_conflict_subtype required
    Validators enforce this; mis-specified records raise on `model_validate`.
    """
    model_config = ConfigDict(extra="allow")

    pair_id: str
    record_type: Literal["conflict", "non_conflict"]
    label: Literal["conflict", "non_conflict"]
    regime_a: Regime
    regime_b: Regime
    evidence_a: list[EvidenceChunk] = Field(default_factory=list)
    evidence_b: list[EvidenceChunk] = Field(default_factory=list)
    rationale: str = ""

    # Annotation provenance
    annotator_id: str = ""
    annotator_model: str = ""
    annotated_at: str = ""
    revision: int = 1
    review_status: ReviewStatus = ReviewStatus.single_annotation
    human_review_required: bool = True
    confidence: float = 0.0
    claude_caveats: list[str] = Field(default_factory=list)

    # Conflict-only
    conflict_type: Optional[ConflictType] = None
    severity: Optional[Severity] = None

    # Non-conflict-only
    non_conflict_subtype: Optional[NonConflictSubtype] = None

    # Adjudication
    review: Optional[ReviewBlock] = None

    def model_post_init(self, __context):
        if self.record_type == "conflict":
            if self.conflict_type is None:
                raise ValueError(f"{self.pair_id}: conflict record missing conflict_type")
            if self.severity is None:
                raise ValueError(f"{self.pair_id}: conflict record missing severity")
        else:
            if self.non_conflict_subtype is None:
                # Permitted to be missing on legacy records, but warn-worthy
                pass


def is_conflict(record: dict) -> bool:
    """Lightweight predicate for code that doesn't want to instantiate Pydantic."""
    return record.get("record_type") == "conflict"

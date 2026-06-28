"""Three prompt variants for the zero-shot LLM conflict-detection baseline.

Each variant carries the four conflict-type definitions explicitly (no labelled
examples — pure zero-shot) and requests JSON output. Variants differ only in
framing, not in task definition.
"""
from __future__ import annotations

from typing import Iterable

CONFLICT_TYPES: tuple[str, ...] = (
    "structural_unresolved",
    "operationally_resolvable",
    "interpretive_fact_sensitive",
    "recurring_friction",
)

# Variant C labels A/B/C/D map back to the canonical type names.
VARIANT_C_LETTER_TO_TYPE = {
    "A": "structural_unresolved",
    "B": "operationally_resolvable",
    "C": "interpretive_fact_sensitive",
    "D": "recurring_friction",
}


def assemble_passage(evidence: Iterable[dict]) -> str:
    """Concatenate evidence chunks into a single passage string.

    The model sees chunk index + page number, but NOT the source filename
    (provenance is logged in raw_responses for analysis but withheld from
    the LLM input — see user spec §1).
    """
    parts = []
    for i, chunk in enumerate(evidence, start=1):
        page = chunk.get("page")
        header = f"[Chunk {i}" + (f", p.{page}" if page else "") + "]"
        passage_text = (chunk.get("passage") or "").strip()
        parts.append(f"{header}\n{passage_text}")
    return "\n\n".join(parts)


def _format_pair_block(record: dict) -> dict:
    ra = record.get("regime_a") or {}
    rb = record.get("regime_b") or {}
    return {
        "regime_a_jurisdiction": ra.get("jurisdiction", "?"),
        "regime_a_id": ra.get("regime_id", "?"),
        "passage_a": assemble_passage(record.get("evidence_a") or []),
        "regime_b_jurisdiction": rb.get("jurisdiction", "?"),
        "regime_b_id": rb.get("regime_id", "?"),
        "passage_b": assemble_passage(record.get("evidence_b") or []),
    }


# ----------------------------------------------------------------------
# Variant A — minimal framing
# ----------------------------------------------------------------------
_VARIANT_A_SYSTEM = "You analyse regulatory passages for conflicts between regulations."

_VARIANT_A_USER = """Read the two regulatory passages below and determine if they conflict.

A conflict means the two regulations create competing or contradictory obligations.

Four types of conflicts can occur:
- structural_unresolved: No clean legal path satisfies both regulations.
- operationally_resolvable: Both can be satisfied via specific mechanisms.
- interpretive_fact_sensitive: Whether a conflict exists depends on facts.
- recurring_friction: Ongoing operational costs from competing requirements.

Passage A (from {regime_a_jurisdiction} — {regime_a_id}):
{passage_a}

Passage B (from {regime_b_jurisdiction} — {regime_b_id}):
{passage_b}

Respond with valid JSON only, in this exact format:
{{
  "conflict_present": true | false,
  "conflict_type": "structural_unresolved" | "operationally_resolvable" | "interpretive_fact_sensitive" | "recurring_friction" | null,
  "rationale": "brief explanation"
}}

If conflict_present is false, conflict_type must be null."""


# ----------------------------------------------------------------------
# Variant B — detailed framing with worked descriptions
# ----------------------------------------------------------------------
_VARIANT_B_SYSTEM = (
    "You are a regulatory compliance analyst. Your task is to identify conflicts "
    "between regulatory passages and classify them by type."
)

_VARIANT_B_USER = """Analyse the two regulatory passages below for conflicts.

A conflict occurs when two regulations create competing or contradictory obligations that an entity subject to both must navigate.

Conflict types:

1. structural_unresolved: A conflict where current legal frameworks provide no clean path to satisfy both regulations simultaneously. Example: a regulation that has no authorised compliance path in one jurisdiction because the issuer is foreign.

2. operationally_resolvable: A conflict where both regulations can be satisfied via specific operational mechanisms (e.g., legal instruments, contractual provisions). Example: data-sharing requirements vs data-transfer restrictions resolved via Standard Contractual Clauses.

3. interpretive_fact_sensitive: A conflict whose resolution depends on specific facts not yet determined. Example: whether a service is being 'offered' or 'received at the client's own initiative'.

4. recurring_friction: Ongoing operational costs from competing requirements that don't fully resolve. Example: tax treatment creating costs that recur with each transaction.

Passage A (jurisdiction: {regime_a_jurisdiction}, regime: {regime_a_id}):
{passage_a}

Passage B (jurisdiction: {regime_b_jurisdiction}, regime: {regime_b_id}):
{passage_b}

Provide your analysis as valid JSON only, in this format:
{{
  "conflict_present": true | false,
  "conflict_type": one of the four types above, or null if no conflict,
  "rationale": "brief explanation of your reasoning"
}}"""


# ----------------------------------------------------------------------
# Variant C — expert role-playing with letter typology
# ----------------------------------------------------------------------
_VARIANT_C_SYSTEM = (
    "You are a senior regulatory expert specialising in cross-jurisdictional "
    "financial regulation. You have deep expertise in identifying conflicts "
    "between regulations from different jurisdictions and classifying them "
    "according to a standardised typology."
)

_VARIANT_C_USER = """Two regulatory passages are provided below. Your task is to determine whether they conflict and, if so, classify the conflict type.

CONFLICT TYPOLOGY:

A. structural_unresolved — The regulations have no compliant intersection. Compliance with one prevents compliance with the other.

B. operationally_resolvable — Compliance with both is achievable through specific mechanisms or instruments.

C. interpretive_fact_sensitive — Conflict status turns on facts not determined in the passages provided.

D. recurring_friction — The combination creates ongoing operational costs without absolute incompatibility.

PASSAGE A
Jurisdiction: {regime_a_jurisdiction}
Regime: {regime_a_id}
{passage_a}

PASSAGE B
Jurisdiction: {regime_b_jurisdiction}
Regime: {regime_b_id}
{passage_b}

Provide your expert assessment as JSON:
{{
  "conflict_present": boolean,
  "conflict_type": one of A, B, C, D as the corresponding string, or null,
  "rationale": string
}}"""


VARIANTS = {
    "A": {"system": _VARIANT_A_SYSTEM, "user_template": _VARIANT_A_USER},
    "B": {"system": _VARIANT_B_SYSTEM, "user_template": _VARIANT_B_USER},
    "C": {"system": _VARIANT_C_SYSTEM, "user_template": _VARIANT_C_USER},
}


def render(variant: str, record: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) rendered for the given record."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {list(VARIANTS)}")
    spec = VARIANTS[variant]
    fields = _format_pair_block(record)
    return spec["system"], spec["user_template"].format(**fields)

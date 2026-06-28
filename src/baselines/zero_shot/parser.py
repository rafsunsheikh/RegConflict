"""Response parsing for the zero-shot LLM baseline.

Strategy:
  1. Strip common wrappers from the raw response (LaTeX \\boxed{}, markdown
     code fences, leading/trailing whitespace).
  2. Locate the JSON object (first balanced { ... } block).
  3. Parse and validate against the schema.
  4. For Variant C: map bare letters (A/B/C/D) to the canonical type names.
  5. If the model returned a substring or fuzzy match, accept if EXACTLY one
     canonical type name appears in the value. Reject ('multiple_types_matched')
     if more than one appears.

Returns ParsedPrediction with parsing_status="success" or "parsing_failure"
plus a failure_reason for error analysis.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from src.baselines.zero_shot.prompts import CONFLICT_TYPES, VARIANT_C_LETTER_TO_TYPE

# Failure reason taxonomy (logged to predictions for error analysis)
FAILURE_REASONS = (
    "no_json_block_found",
    "invalid_json",
    "schema_missing_fields",
    "schema_wrong_type",
    "multiple_types_matched",
    "unknown_conflict_type",
    "type_set_without_conflict",
    "all_retries_failed",
)


@dataclass
class ParsedPrediction:
    success: bool
    conflict_present: bool | None = None
    conflict_type: str | None = None
    rationale: str | None = None
    failure_reason: str | None = None
    failure_detail: str | None = None
    extracted_json_str: str | None = None  # for debugging
    raw_response: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


_MD_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?", re.MULTILINE)
_BOXED_RE = re.compile(r"\\boxed\{([\s\S]*?)\}\s*$")


def _strip_wrappers(text: str) -> str:
    """Remove markdown code fences, \\boxed{} wrapping, and outer whitespace."""
    text = text.strip()
    # \boxed{...}
    m = _BOXED_RE.search(text)
    if m:
        text = m.group(1).strip()
    # ```json ... ```
    text = _MD_FENCE_RE.sub("", text)
    text = text.replace("```", "").strip()
    return text


def _find_json_block(text: str) -> str | None:
    """Locate the first balanced {...} block in `text`. Returns None if not found."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _normalise_conflict_type(value, variant: str) -> tuple[str | None, str | None]:
    """Normalise a model's `conflict_type` field to a canonical type or None.

    Returns (canonical_type_or_None, failure_reason_or_None).
    Cases:
      - value is None or "null"   → (None, None)
      - exact match (case-insensitive) to a canonical type → (canonical, None)
      - Variant C bare letter (A/B/C/D) → mapped canonical type
      - substring of value contains EXACTLY one canonical type name → (canonical, None)
      - substring contains >1 canonical type names → (None, "multiple_types_matched")
      - anything else → (None, "unknown_conflict_type")
    """
    if value is None or (isinstance(value, str) and value.strip().lower() in {"null", "none", ""}):
        return None, None
    if not isinstance(value, str):
        return None, "schema_wrong_type"
    v = value.strip()
    v_lower = v.lower()

    # Exact match
    for t in CONFLICT_TYPES:
        if v_lower == t.lower():
            return t, None

    # Variant C bare letter (case insensitive, may be "A" or "A." or "Option A")
    if variant == "C":
        letter_match = re.search(r"\b([ABCD])\b", v.upper())
        if letter_match:
            letter = letter_match.group(1)
            if letter in VARIANT_C_LETTER_TO_TYPE:
                return VARIANT_C_LETTER_TO_TYPE[letter], None

    # Substring containing exactly one canonical type name
    matches = [t for t in CONFLICT_TYPES if t.lower() in v_lower]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, "multiple_types_matched"

    return None, "unknown_conflict_type"


def parse(raw_response: str, variant: str) -> ParsedPrediction:
    """Parse a model's response. Always returns a ParsedPrediction; check .success."""
    if not raw_response or not raw_response.strip():
        return ParsedPrediction(success=False, failure_reason="no_json_block_found",
                                raw_response=raw_response or "")

    stripped = _strip_wrappers(raw_response)
    json_block = _find_json_block(stripped)
    if json_block is None:
        return ParsedPrediction(success=False, failure_reason="no_json_block_found",
                                raw_response=raw_response)

    try:
        obj = json.loads(json_block)
    except json.JSONDecodeError as e:
        return ParsedPrediction(
            success=False,
            failure_reason="invalid_json",
            failure_detail=str(e),
            extracted_json_str=json_block,
            raw_response=raw_response,
        )

    # Required fields
    if "conflict_present" not in obj:
        return ParsedPrediction(
            success=False,
            failure_reason="schema_missing_fields",
            failure_detail="missing conflict_present",
            extracted_json_str=json_block,
            raw_response=raw_response,
        )

    # conflict_present must be bool (some models emit "true"/"false" as strings)
    cp_raw = obj["conflict_present"]
    if isinstance(cp_raw, bool):
        conflict_present = cp_raw
    elif isinstance(cp_raw, str) and cp_raw.strip().lower() in {"true", "false"}:
        conflict_present = cp_raw.strip().lower() == "true"
    else:
        return ParsedPrediction(
            success=False,
            failure_reason="schema_wrong_type",
            failure_detail=f"conflict_present must be bool, got {cp_raw!r}",
            extracted_json_str=json_block,
            raw_response=raw_response,
        )

    # conflict_type — accept null or string; normalise via variant-aware logic
    ct_raw = obj.get("conflict_type")
    conflict_type, ct_failure = _normalise_conflict_type(ct_raw, variant)
    if ct_failure:
        return ParsedPrediction(
            success=False,
            failure_reason=ct_failure,
            failure_detail=f"conflict_type={ct_raw!r}",
            extracted_json_str=json_block,
            raw_response=raw_response,
        )

    # Cross-field invariant: if conflict_present=False, conflict_type must be None
    if not conflict_present and conflict_type is not None:
        return ParsedPrediction(
            success=False,
            failure_reason="type_set_without_conflict",
            failure_detail=f"conflict_present=False but conflict_type={conflict_type!r}",
            extracted_json_str=json_block,
            raw_response=raw_response,
        )

    rationale = obj.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        rationale = str(rationale)

    return ParsedPrediction(
        success=True,
        conflict_present=conflict_present,
        conflict_type=conflict_type,
        rationale=rationale,
        extracted_json_str=json_block,
        raw_response=raw_response,
    )

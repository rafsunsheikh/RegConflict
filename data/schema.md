# JSONL record schema

Every record in `conflicts/{train,val,test,gold_iaa}.jsonl` follows the same JSON schema. Each line is one regime-pair record (a *labelled pair*). The same record shape is used for `conflict` and `non_conflict` records; a discriminator field (`record_type`) tells you which is which, and type-specific fields are populated accordingly.

## Top-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `pair_id` | string | yes | Stable identifier, format `{record_type}:{16-hex-chars}`. Globally unique across splits. |
| `record_type` | enum | yes | One of `"conflict"`, `"non_conflict"`. Discriminator for the type-specific block below. |
| `label` | enum | yes | Mirrors `record_type` for convenience: `"conflict"` or `"non_conflict"`. |
| `regime_a` | object | yes | First regime in the pair. See [Regime object](#regime-object). |
| `regime_b` | object | yes | Second regime in the pair. Same schema as `regime_a`. |
| `evidence_a` | array | yes | Evidence chunks for `regime_a` — the passages an annotator and a model see for that side. See [Evidence chunk](#evidence-chunk). |
| `evidence_b` | array | yes | Same shape as `evidence_a`, for `regime_b`. |
| `rationale` | string | yes (conflict); yes (non_conflict) | Free-text human-readable justification for the label. For conflicts: why these two regimes conflict and how. For non-conflicts: why these two regimes do not conflict despite topical proximity. |
| `annotator_id` | string | yes | Opaque ID of the primary annotator. The IAA subset additionally has a `review` block (see below). |
| `annotator_model` | string | yes | If labelling was assisted by an LLM, the model identifier and provenance. For purely human labels, `"human"`. |
| `annotated_at` | string (ISO-8601) | yes | UTC timestamp of the initial annotation. |
| `revision` | integer | yes | 1-indexed revision count; increments on substantive re-label. |
| `review_status` | enum | yes | `"single_annotation"`, `"reviewed_accept"`, `"reviewed_revise"`, `"reviewed_reject"`. |
| `human_review_required` | boolean | yes | True if the record needs (or needed) expert review beyond the initial pass. |
| `confidence` | float | yes | Annotator's reported confidence in the label, ∈ [0, 1]. |
| `claude_caveats` | array of strings | yes (may be `[]`) | Annotator-noted caveats, edge-case observations, or reasons the label is uncertain. |
| `review` | object | optional | Present on records that went through a second-pass review or adjudication. See [Review block](#review-block). |

## Conflict-type-specific fields

Present only when `record_type == "conflict"`:

| Field | Type | Required (on conflict) | Description |
|---|---|---|---|
| `conflict_type` | enum | yes | One of `"structural_unresolved"`, `"operationally_resolvable"`, `"interpretive_fact_sensitive"`, `"recurring_friction"`. See [`annotations/decision_rubric.md`](annotations/decision_rubric.md). |
| `severity` | enum | yes | One of `"low"`, `"medium"`, `"high"`. Annotator's judgement of how serious the conflict is for an entity subject to both regimes. |

## Non-conflict-type-specific fields

Present only when `record_type == "non_conflict"`:

| Field | Type | Required (on non_conflict) | Description |
|---|---|---|---|
| `non_conflict_subtype` | enum | yes | One of `"aligned_complementary"`, `"different_domains"`, `"superficially_similar_but_disjoint"`, `"same_jurisdiction_no_overlap"`. Free-form taxonomy of *why* the pair is not a conflict; **not** part of the evaluation task. Provided for analysis and for hard-negative selection during training. |

## Regime object

```jsonc
{
  "regime_id": "EU_MICA",          // canonical identifier (uppercase snake_case)
  "jurisdiction": "EU",            // one of: "Australia", "EU", "Singapore", "International"
  "issuing_body": "EUR-Lex",       // organisation/source (may be "?" for older records)
  "short_name": "EU_MICA"          // display name; equal to regime_id by default
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `regime_id` | string | yes | Canonical regime identifier. Must match an entry in `corpus/document_inventory.csv`. |
| `jurisdiction` | enum | yes | One of the four covered values. International standards-setting bodies (FATF, BIS, IOSCO) appear as `"International"`. |
| `issuing_body` | string | yes | Free-text source organisation. `"?"` is permitted for legacy records where this was not captured. |
| `short_name` | string | yes | Display name; commonly equal to `regime_id`. |

## Evidence chunk

`evidence_a` and `evidence_b` are arrays of evidence chunks. Each chunk is a passage drawn from a corpus document that supports the rationale for the regime side.

```jsonc
{
  "chunk_id": "32afd37cade34f84::34b::8d30",   // stable chunk identifier
  "source_doc": "d424.pdf",                    // filename in corpus/
  "page": 148,                                  // 1-indexed page number (PDF); may be null
  "passage": "11 Where a bank according to..."  // verbatim text of the chunk
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `chunk_id` | string | yes | Stable identifier composed of `{doc_hash}::{position}::{nonce}`. |
| `source_doc` | string | yes | Filename within `corpus/` (or in `corpus/tier3_metadata_only/manifest.csv` for withheld documents). |
| `page` | integer or null | yes | PDF page number (1-indexed) where the passage appears; `null` for non-paginated sources. |
| `passage` | string | yes | Verbatim text of the chunk. Whitespace and line breaks preserved as extracted from the source. |

## Review block (when present)

Records that have gone through adjudication carry an additional `review` object:

```jsonc
{
  "review": {
    "reviewer_id": "10116",                    // opaque reviewer ID
    "reviewed_at": "2026-06-18T13:26:57Z",     // ISO-8601 UTC
    "decision": "accept",                       // "accept", "revise", "reject"
    "notes": "",                                // free-text reviewer notes
    "original_conflict_type": "structural_unresolved",  // pre-review value, if changed
    "original_severity": "high",                // pre-review value
    "original_review_status": "single_annotation"
  }
}
```

The `original_*` fields preserve pre-review values so revisions are auditable. If a record's label was changed during review, the post-review values appear at the top level and the original values appear inside `review`.

## What the task expects

The benchmark defines two prediction tasks layered on each record:

1. **Binary conflict detection.** Given (`regime_a`, `regime_b`, `evidence_a`, `evidence_b`), predict `record_type == "conflict"` (boolean).
2. **Four-class typology classification.** *Conditional on* a positive binary prediction, predict the `conflict_type` ∈ {`structural_unresolved`, `operationally_resolvable`, `interpretive_fact_sensitive`, `recurring_friction`}.

Inputs visible to models at inference time: `regime_a.regime_id`, `regime_a.jurisdiction`, `evidence_a[*].passage` (concatenated per side), and the same for `regime_b`. Inputs **not** visible: `rationale`, `annotator_*`, `review`, `confidence`, `non_conflict_subtype`, and the gold `conflict_type` / `severity` for conflict records. These are available to dataset users but withheld from the model input pipeline by `src/eval/benchmark/`.

## Conventions

- All timestamps are ISO-8601 UTC strings.
- All identifiers (`pair_id`, `chunk_id`, `regime_id`) are stable across releases within a major version; cross-version stability for `pair_id` is preserved unless noted in `CHANGELOG.md`.
- Empty values are represented as `""` (string) or `null` (object) rather than missing keys, so downstream parsers can rely on key presence.
- The four `jurisdiction` enum values are deliberately conservative — international standards-setting bodies (FATF, BIS, IOSCO) all appear as `"International"`, and country-level identifiers below the jurisdiction (e.g., AU state-level regulation, EU member-state regulation) are not represented in this schema.

## Where the schema is implemented

The Python representation lives in `src/eval/benchmark/schema.py`. A Pydantic v2 model is provided there; using it instead of raw dict access is recommended for downstream code that wants validation on read.

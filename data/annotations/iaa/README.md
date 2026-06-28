# `data/annotations/iaa/` — Inter-annotator agreement artefacts

This directory holds the raw IAA data used to compute the canonical κ disclosure in §4.5 of the paper (Cohen's κ = 0.6 on the four-class typology task, n = 30 conflict pairs, substantial agreement on the Landis–Koch scale).

## Files

| File | Contents | Status in v1.0 |
|---|---|---|
| `annotator1_labels.jsonl` | Primary annotator's typology labels on the 30 IAA pairs | **[FILL: merge from offline source]** |
| `annotator2_labels.jsonl` | Second annotator's blind typology labels on the same 30 pairs | **[FILL: merge from offline source]** |
| `adjudicated.jsonl` | Post-adjudication gold labels (the consensus typology after joint review of disagreements) | **[FILL: merge from offline source]** |
| `calibration_pairs.jsonl` | Pairs used in the pre-blind-pass rubric-calibration round (disjoint from the 30-pair IAA sample) | **[FILL: optional but recommended for full transparency]** |

## Required schema

Each line of `annotator1_labels.jsonl` and `annotator2_labels.jsonl` should be a JSON object with at minimum:

```json
{
  "pair_id": "conflict:abcd1234ef5678901234567890abcdef",
  "conflict_type": "operationally_resolvable"
}
```

Optional fields preserved for downstream analysis (e.g., per-pair disagreement vignettes):

```json
{
  "pair_id": "...",
  "annotator_id": "primary" | "second_2026" | etc.,
  "conflict_type": "operationally_resolvable",
  "severity": "medium",
  "confidence": 0.85,
  "rationale": "...",
  "notes": "..."
}
```

The `compute_iaa.py` script reads only `pair_id` and `conflict_type` from each file. All other fields are preserved for traceability.

## How to populate from the existing offline data

If your 30-pair second-annotator pass is in a single combined file (one record per pair with primary's `conflict_type` and a `blind_review` block containing the second annotator's `conflict_type`), the simplest split is:

```bash
python3 << 'EOF'
import json
from pathlib import Path

src = Path("path/to/your/30_pair_iaa_combined.jsonl")
out_dir = Path("data/annotations/iaa")
out_dir.mkdir(parents=True, exist_ok=True)

records = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

with (out_dir / "annotator1_labels.jsonl").open("w") as f:
    for r in records:
        f.write(json.dumps({
            "pair_id": r["pair_id"],
            "conflict_type": r["conflict_type"],   # primary
            "annotator_id": "primary",
        }) + "\n")

with (out_dir / "annotator2_labels.jsonl").open("w") as f:
    for r in records:
        br = r["blind_review"]
        f.write(json.dumps({
            "pair_id": r["pair_id"],
            "conflict_type": br["conflict_type"],  # second annotator
            "annotator_id": br.get("reviewer_id", "second"),
            "notes": br.get("notes", ""),
        }) + "\n")
EOF
```

After populating, verify the κ computation:

```bash
python scripts/compute_iaa.py
```

Expected output:

```
Loaded annotator1 labels:  30 records
Loaded annotator2 labels:  30 records
Pair-id alignment: 30 matched
Four-class typology task (N = 30):
  raw agreement:  XX/30 (XX.X%)
  Cohen's κ:      0.6XXX  (substantial — Landis–Koch)
  95% bootstrap CI: [...]
```

The expected raw-agreement count for κ ≈ 0.6 with reasonably balanced primary marginals is approximately 22–24 of 30 (73–80% raw agreement); the exact count depends on the second annotator's confusion structure.

## Once populated, this README's role

After the v1.0 release ships, the `[FILL]` rows in the table above should become `present` markers, and this README's "how to populate" section can be archived in `CHANGELOG.md` rather than kept inline.

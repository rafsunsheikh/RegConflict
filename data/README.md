# `data/` — RegConflict data assets

This directory holds everything you need to use RegConflict as a benchmark: the labelled pairs, the source corpus, the annotation artefacts, and the datasheet.

## Layout

```
data/
├── README.md                 ← this file
├── schema.md                 ← JSONL record schema for the labelled pairs
├── datasheet.md              ← Gebru et al. (2021) datasheet
│
├── conflicts/                ← labelled regime-pair records (one JSONL per split)
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── gold_iaa.jsonl
│
├── splits/
│   └── split_metadata.json   ← seed, proportions, stratification, per-split counts
│
├── corpus/
│   ├── document_inventory.csv          ← all 247 documents + license metadata
│   ├── tier1_redistributable/          ← 166 raw PDFs/DOCXs, full content, open license
│   ├── tier2_conditional/              ← 26 raw PDFs, academic-use redistribution
│   └── tier3_metadata_only/            ← 55 documents, metadata + fetch URLs only
│       └── manifest.csv
│
├── extracted/                ← extracted plain text per source document
│   ├── Australia/<source>/<stem>__<sha16>.json    ← one file per Tier 1/2 doc
│   ├── EU/<source>/...
│   ├── International/<source>/...
│   └── Singapore/<source>/...
│       (Tier 3 sources — MAS-*, IRAS, FATF — are NOT shipped here; users
│       regenerate locally via `python src/tools/chunk_corpus.py --tier 3`)
│
├── chunks/                   ← segmented passages per source document
│   ├── Australia/<source>/<stem>__<sha16>.jsonl   ← JSONL, one chunk per line
│   ├── EU/<source>/...
│   ├── International/<source>/...
│   └── Singapore/<source>/...
│       (same Tier 3 caveat as extracted/)
│
└── annotations/              ← annotation artefacts and IAA records
    ├── guidelines.md         ← annotator instructions
    ├── decision_rubric.md    ← per-class typology decision rules
    └── iaa/
        ├── annotator1_labels.jsonl     ← primary annotator labels on IAA subset
        ├── annotator2_labels.jsonl     ← second annotator labels on IAA subset
        └── adjudicated.jsonl           ← post-adjudication labels (= gold_iaa.jsonl)
```

## At a glance — what's in the labelled splits

| Split | Records | Conflict | Non-conflict |
|---|---:|---:|---:|
| `train.jsonl` | 497 | 65 | 432 |
| `val.jsonl` | 107 | 14 | 93 |
| `test.jsonl` | 107 | 14 | 93 |
| `gold_iaa.jsonl` | 13 | 8 | 5 |
| **Total** | **724** | **101** | **623** |

The benchmark contains **724 labelled regime pairs across train/val/test/gold-IAA splits, of which 101 are conflicts and 623 are non-conflicts.** This is the load-bearing number. *(If you've seen "approximately 250 conflict pairs" in earlier paper drafts, that figure was imprecise — it conflated "labelled pairs" with "conflict pairs"; the actual conflict count is 101.)*

Splits are stratified by `conflict_type` (5 strata: 4 conflict types + `non_conflict`) and partitioned at the regime-pair level to prevent leakage.

The **30-pair gold IAA subset** (used for inter-annotator agreement measurement; Cohen's κ = 0.6 on the four-class typology task) is a separately-held random sample of 30 conflict pairs with both annotators' independent typology labels. Its data and adjudicated gold labels live in [`annotations/iaa/`](annotations/iaa/); the κ computation is reproducible via `scripts/compute_iaa.py`. The 13-record `gold_iaa.jsonl` in this directory is the legacy v0.9 held-out evaluation set; v1.0 expects this file to be replaced with the 30 adjudicated records from `annotations/iaa/adjudicated.jsonl` — see [`annotations/iaa/README.md`](annotations/iaa/README.md) for the merge protocol.

Per-conflict-type counts on the test split:

- `structural_unresolved`: 2
- `operationally_resolvable`: 8
- `interpretive_fact_sensitive`: 2
- `recurring_friction`: 2

Three of the four classes are at n=2 in test — read per-class metrics as directional, not precise.

## At a glance — what's in the corpus

The corpus is **247 regulatory documents from 18 source organisations**, packaged by licence tier:

| Tier | Documents | Treatment |
|---|---:|---|
| Tier 1 — open licensing | 166 | Redistributed in full under each source's open licence (CC BY 4.0, Crown Copyright OGL, EU Reuse Decision 2011/833/EU, equivalents) |
| Tier 2 — academic-use redistribution | 26 | Redistributed under each source's academic-use terms with attribution |
| Tier 3 — metadata-only | 55 | Title, jurisdiction, issuing body, source URL, SHA-256 included; document content **not** redistributed. Use `src/tools/fetch_tier3.py` to retrieve from official sources. |

Per-source-organisation breakdown and verbatim license-clause quotations are in [`../LICENSES/SOURCE_DOCUMENT_LICENSES.md`](../LICENSES/SOURCE_DOCUMENT_LICENSES.md).

Tier 3 sources: MAS (34 documents), FATF (19), IRAS (2).

## The cleaned corpus — `extracted/` and `chunks/`

`extracted/` and `chunks/` hold the output of the ingestion pipeline (Steps 1–3 of `src/pipeline/`):

- **`extracted/<jurisdiction>/<source>/<stem>__<sha16>.json`** — One JSON per source document with full extracted text, organised by page. Used by users who want to run their own chunking strategy, OCR comparisons, or full-document analysis.
- **`chunks/<jurisdiction>/<source>/<stem>__<sha16>.jsonl`** — One JSONL per source document; each line is a chunk record with stable `chunk_id`, page provenance, citation, parent citation, document type, and the canonical chunk `text`. This is what the labelled records' `evidence_a` and `evidence_b` passages were drawn from, and what retrieval-augmented baselines should index over.

### Tier coverage

The cleaned corpus is shipped for **Tier 1 + Tier 2 documents only** (192 of 247). The 55 Tier 3 documents (MAS, FATF, IRAS) cannot be redistributed under their source licenses, and that restriction extends to substantial derivative works like full extracted text and chunks. The labelled records still reference Tier 3 chunks (via the `evidence_a[*].passage` text embedded in the JSONL records, which falls under academic-quotation fair-use), but the full per-document extracted/chunks files for Tier 3 are not on disk in the release.

To regenerate the Tier 3 cleaned corpus locally after fetching the source PDFs:

```bash
# Step 1 — fetch Tier 3 PDFs from their source URLs (see fetch_tier3.py for FATF caveats)
python src/tools/fetch_tier3.py

# Step 2 — run the ingestion pipeline on the fetched PDFs
python src/tools/chunk_corpus.py --tier 3
```

After Step 2, `extracted/` and `chunks/` contain the full 247-document corpus. The chunking strategy is pinned in `src/pipeline/config.py` (`CHUNK_TARGET_TOKENS=200`, `CHUNK_MAX_TOKENS=400`, no overlap, tiktoken cl100k tokeniser) so Tier 3 chunks regenerated locally are byte-identical to what the release authors produced.

### Anonymisation

Each extracted/chunk record carries a `source_path` field. In the released files, this is a repo-relative path like `data/corpus/tier1_redistributable/APRA/141121-RIS-RFCs.pdf` — never an absolute path on the release authors' machines. The `chunk_corpus.py` tool maintains this invariant on regenerated Tier 3 files.

### Chunk record schema

```jsonc
{
  "chunk_id": "74ae1e75a88e3045::3fb::8ae7",   // stable chunk identifier
  "doc_id": "74ae1e75a88e3045",                 // 16-hex SHA-256 prefix of source file
  "jurisdiction": "Australia",
  "issuing_body": "APRA",
  "document_type": "paragraph",                 // canonical citation unit per source
  "filename": "141121-RIS-RFCs.pdf",
  "source_path": "data/corpus/tier1_redistributable/APRA/141121-RIS-RFCs.pdf",
  "document_hash": "74ae1e75a88e3045",
  "citation": "Regulation Impact Statement",    // human-readable citation
  "parent_citation": null,
  "kind": "section",                            // structural unit
  "page_start": null,
  "page_end": 4,
  "retrieval_date": "2026-05-23",
  "effective_date": null,
  "source_url": null,
  "text": "Regulation Impact Statement\n...",   // the chunk content
  "n_tokens": 213,                              // tiktoken cl100k count
  "regime_tags": [],                            // populated by future tag pass
  "topic_tags": [],
  "applicability_tags": []
}
```

## How to use this data

### Load a split

```python
import json
from pathlib import Path

records = [json.loads(l) for l in
           Path("data/conflicts/test.jsonl").read_text().splitlines() if l.strip()]
print(f"loaded {len(records)} pairs; "
      f"{sum(1 for r in records if r['record_type']=='conflict')} are conflicts")
```

The full record schema is documented in [`schema.md`](schema.md).

### Reconstruct the Tier 3 documents

After downloading the repository, the Tier 3 manifest at `corpus/tier3_metadata_only/manifest.csv` lists each withheld document's source URL and SHA-256 checksum. The fetch tool (Phase 3) automates retrieval:

```bash
python src/tools/fetch_tier3.py --output data/corpus/tier3_metadata_only/
```

Each retrieved document is verified against the published SHA-256 before being written to disk; if a source URL has drifted, the tool surfaces the failure with the expected hash so you can locate the document manually.

### Reproduce the splits from raw labels

```bash
python src/tools/build_splits.py --seed 42 --output data/splits/
```

The split tool (Phase 3) reads `annotations/conflicts.jsonl` plus `annotations/non_conflicts.jsonl`, applies the stratification described in `splits/split_metadata.json`, and writes the four split files. Re-running with the same seed produces byte-identical outputs.

### Verify dataset integrity

```bash
python src/tools/verify_corpus.py
```

The verify tool checks that every file referenced by `corpus/document_inventory.csv` is present (or correctly marked metadata-only), that all SHA-256 hashes match, and that every record in `conflicts/*.jsonl` references valid `regime_id`s present in the inventory.

## Versioning

The data is versioned together with the repository. The v1.0.0 release of RegConflict pins the exact label set and split assignment used by the paper's reported numbers; downstream releases (v1.1, v2.0, …) may revise labels, expand jurisdictions, or refine the typology. See [`../CHANGELOG.md`](../CHANGELOG.md).

## Citing the data

Please cite both the paper and the dataset deposit when you use this data. See [`../CITATION.cff`](../CITATION.cff) for the canonical citation, and [`../LICENSES/`](../LICENSES/) for the full layered licensing scheme.

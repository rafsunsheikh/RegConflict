# Annotations License — Creative Commons Attribution 4.0 International (CC BY 4.0)

All annotations, labels, and derived materials in this repository are released under the **Creative Commons Attribution 4.0 International** license (CC BY 4.0).

Canonical license text and machine-readable license deed:
<https://creativecommons.org/licenses/by/4.0/>
SPDX identifier: `CC-BY-4.0`

## What this covers

The CC BY 4.0 license applies to all *derived annotations* layered on top of the source regulatory documents, including:

- The four-class regulatory conflict typology (`structural_unresolved`, `operationally_resolvable`, `interpretive_fact_sensitive`, `recurring_friction`) and the decision rubric for assigning labels (`data/annotations/decision_rubric.md`).
- The annotation guidelines (`data/annotations/guidelines.md`).
- The labelled conflict pairs in `data/conflicts/` (train, val, test, gold IAA).
- The inter-annotator agreement artefacts in `data/annotations/iaa/`.
- The datasheet (`data/datasheet.md`) following Gebru et al. (2021).
- The document inventory metadata in `data/corpus/document_inventory.csv` (the metadata describing each source document — not the source documents themselves).
- The split metadata in `data/splits/split_metadata.json`.

## What this does NOT cover

CC BY 4.0 does **not** apply to:

- **The source regulatory documents themselves.** Each retains its original licensing as published by the issuing authority. See [`SOURCE_DOCUMENT_LICENSES.md`](SOURCE_DOCUMENT_LICENSES.md) for per-source classification (Tier 1 / Tier 2 / Tier 3) and [`/ATTRIBUTIONS.md`](../ATTRIBUTIONS.md) for required source-level attribution strings.
- **Code.** Released under MIT — see [`LICENSE-CODE.md`](LICENSE-CODE.md).

## Attribution requirements

Under CC BY 4.0, you may share and adapt the annotations for any purpose, including commercial use, provided you give appropriate credit, indicate any changes you made, and provide a link back to the license. A minimal attribution that satisfies these requirements:

> RegConflict annotations (CC BY 4.0). Released alongside [paper citation — see `/CITATION.cff`]. Source: <repository URL>.

If you also redistribute or derive from any of the source regulatory documents covered in this release, the per-source attribution strings in [`/ATTRIBUTIONS.md`](../ATTRIBUTIONS.md) must additionally be preserved.

## Summary of permissions (informational, not a substitute for the license)

| You may | Under condition |
|---|---|
| Share — copy and redistribute the annotations in any medium or format | Attribution |
| Adapt — remix, transform, and build upon the annotations for any purpose, including commercial | Attribution |
| Combine the annotations with code, data, or other annotations under compatible terms | Attribution |

| You may not | Reason |
|---|---|
| Apply legal terms or technological measures that legally restrict others from doing anything the license permits | CC BY 4.0 forbids "no additional restrictions" |
| Imply endorsement by the original authors of derivative works | CC BY 4.0 imposes a no-endorsement requirement |

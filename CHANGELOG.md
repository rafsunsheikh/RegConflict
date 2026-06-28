# Changelog

All notable changes to RegConflict will be recorded here. The project follows [semantic versioning](https://semver.org/) (MAJOR.MINOR.PATCH). Pre-publication releases use the `0.x.y` prefix.

## [Unreleased]

### Phase 1 — Repository foundation (in progress)

- Initial directory skeleton.
- Top-level `LICENSE` (MIT) and layered licensing scheme under `LICENSES/`.
- `ATTRIBUTIONS.md` per-source attribution strings.
- `CITATION.cff` citation metadata.
- This `CHANGELOG.md`.

### Planned for Phase 2 — Documentation and packaging

- Datasheet for the dataset following Gebru et al. (2021).
- Conflict JSONLs packaged under `data/conflicts/`.
- Source corpus packaged under `data/corpus/tier1_redistributable/`, `tier2_conditional/`, `tier3_metadata_only/`.
- `data/corpus/document_inventory.csv` covering all 247 source documents.
- `data/annotations/guidelines.md` and `data/annotations/decision_rubric.md`.

### Planned for Phase 3 — Code and tools

- Generalisation of hard-coded paths to configuration.
- `src/tools/fetch_tier3.py` — download + SHA-256 verification for Tier 3 documents.
- `src/tools/build_splits.py` — reproduce data splits from raw labels.
- `src/tools/verify_corpus.py` — integrity check on the released package.
- YAML configs under `configs/`.

### Planned for Phase 4 — Tests, notebooks, CI

- Unit and integration tests under `tests/`.
- Reproduction notebooks under `notebooks/`.
- GitHub Actions CI workflows under `.github/workflows/`.

### Planned for Phase 5 — Hosting

- Zenodo deposit (versioned DOI).
- HuggingFace Datasets mirror.
- Anonymous review mirror via anonymous.4open.science (submission window only).

---

## [0.1.0] — Repository skeleton (planned)

First tagged pre-release. Structure and documentation, no data or code yet.

## [1.0.0] — Initial public release (planned, EMNLP 2027)

First non-anonymous release coinciding with paper publication. Includes:

- Four-class typology and decision rubric.
- 724 labelled regime pairs (101 conflicts) across six jurisdictional sources, with train/val/test splits and a separately-held 30-pair gold IAA subset of conflict cases (Cohen's κ = 0.6 on the four-class typology).
- 247-document corpus with per-source licensing tier classification (166 Tier 1 / 26 Tier 2 / 55 Tier 3).
- Baseline implementations: random predictors, zero-shot prompted LLMs at two parameter scales, supervised fine-tuned DeBERTa-v3-base cross-encoder.
- Evaluation harness with bootstrap CIs and paired bootstrap significance testing.
- Documentation: README, datasheet, annotation guidelines, decision rubric, reproducibility instructions.

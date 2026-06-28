# RegConflict

[![DOI](https://zenodo.org/badge/1283199318.svg)](https://doi.org/10.5281/zenodo.21007776)

**A cross-jurisdictional benchmark for regulatory conflict detection.**

RegConflict is an expert-annotated benchmark for the task of detecting conflicts between regulatory regimes from different jurisdictions in financial regulation. It pairs a four-class typology of conflict kinds with ≈250 labelled regime pairs, a curated 247-document multi-jurisdictional regulatory corpus, and reproducible baselines spanning random predictors, zero-shot prompted LLMs, and a supervised fine-tuned cross-encoder.

> ⚠ **Pre-publication.** This repository is the staging area for the public RegConflict v1.0 release accompanying the EMNLP 2027 paper. Data and code packaging are tracked in [`CHANGELOG.md`](CHANGELOG.md); contents are subject to change before the v1.0 tag.

---

## What's here

- **A four-class typology** of regulatory conflicts — `structural_unresolved`, `operationally_resolvable`, `interpretive_fact_sensitive`, `recurring_friction` — distinguished by the resolution mechanism (or absence thereof) available to an entity facing both regimes.
- **724 labelled regime pairs** (101 conflicts) across six jurisdictional sources (Australia, the European Union, Singapore, FATF, BIS, IOSCO), with train / validation / test splits and a separately-held 30-pair gold IAA subset of conflict cases used to measure inter-annotator agreement (Cohen's κ = 0.6 on the four-class typology task — substantial agreement per Landis–Koch).
- **A 247-document regulatory corpus** from 18 source organisations, with per-source three-tier licensing classification (166 redistributable Tier 1; 26 academic-use Tier 2; 55 metadata-only Tier 3 with fetch-from-source scripts).
- **Baseline implementations** for random predictors (majority class, stratified random), zero-shot prompted LLMs at two parameter scales (1.5B DeepSeek-R1-Distill-Qwen, ≈30B GLM-4.7-Flash) with three prompt variants each, and a supervised fine-tuned DeBERTa-v3-base cross-encoder with two classification heads.
- **An evaluation harness** with bootstrap confidence intervals, paired-bootstrap significance testing across baselines, and per-conflict-type / per-jurisdiction / per-severity breakdowns.

---

## Quick start

> 📦 The quick-start code paths below assume the v1.0 packaging layout. During the pre-publication phase, paths under `data/` and `src/` are skeletons; full content lands in Phase 2 of [`CHANGELOG.md`](CHANGELOG.md).

### Install

```bash
git clone https://github.com/OWNER/RegConflict.git    # replace OWNER once public
cd RegConflict
pip install -r requirements.txt
```

### Load the labelled splits

```python
import json
from pathlib import Path

def load_split(name):
    path = Path("data/conflicts") / f"{name}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

train = load_split("train")
val   = load_split("val")
test  = load_split("test")
gold  = load_split("gold_iaa")
print(f"train={len(train)}  val={len(val)}  test={len(test)}  gold_iaa={len(gold)}")
```

### Run the random and majority-class baselines

```bash
python scripts/run_eval.py --baseline majority_class --split test
python scripts/run_eval.py --baseline stratified_random --split test --seeds 100
```

### Reproduce the paper

End-to-end reproduction instructions are in [`docs/reproducibility.md`](docs/reproducibility.md) (forthcoming in Phase 4). The headline numbers — DeBERTa F1 = 0.517, zero-shot GLM F1 = 0.100, zero-shot DeepSeek F1 = 0.135, stratified random F1 = 0.127 — should reproduce bit-for-bit from the v1.0 release given the same random seeds (42–46 for the supervised baseline; 0–99 for stratified random).

---

## Repository layout

```
RegConflict/
├── README.md                 ← this file
├── LICENSE                   ← MIT (code)
├── LICENSES/
│   ├── LICENSE-CODE.md
│   ├── LICENSE-ANNOTATIONS.md      ← CC BY 4.0 (annotations)
│   └── SOURCE_DOCUMENT_LICENSES.md ← per-source three-tier classification
├── ATTRIBUTIONS.md           ← required attribution strings per source
├── CITATION.cff              ← citation metadata (CFF format)
├── CHANGELOG.md
│
├── data/
│   ├── conflicts/            ← labelled pairs: train / val / test / gold_iaa
│   ├── splits/               ← split metadata
│   ├── corpus/               ← 247 source documents, packaged by license tier
│   │   ├── tier1_redistributable/   ← 166 docs, full content
│   │   ├── tier2_conditional/       ← 26 docs, academic-use
│   │   └── tier3_metadata_only/     ← 55 docs, metadata + fetch scripts
│   ├── annotations/          ← guidelines, decision rubric, IAA records
│   └── datasheet.md          ← Gebru et al. (2021) datasheet
│
├── src/
│   ├── eval/                 ← evaluation harness, metrics, bootstrap
│   ├── baselines/            ← random / zero_shot / fine_tuned
│   └── tools/                ← fetch_tier3.py, build_splits.py, verify_corpus.py
│
├── scripts/                  ← CLI entry points (run_eval.py, train_*, etc.)
├── configs/                  ← YAML hyperparameters and prompt variants
├── notebooks/                ← reproduction + exploration notebooks
├── docs/                     ← evaluation methodology, annotation protocol, reproducibility
├── tests/                    ← unit + integration tests
└── .github/                  ← issue templates + CI workflows
```

---

## Licensing

This release uses a layered licensing scheme. Read each licence in full before reuse.

| Material | License | Reference |
|---|---|---|
| **Code** (harness, baselines, tools, tests, configs, notebooks) | **MIT** | [`LICENSE`](LICENSE), [`LICENSES/LICENSE-CODE.md`](LICENSES/LICENSE-CODE.md) |
| **Annotations** (typology, labels, decision rubric, IAA records, datasheet, document inventory metadata) | **CC BY 4.0** | [`LICENSES/LICENSE-ANNOTATIONS.md`](LICENSES/LICENSE-ANNOTATIONS.md) |
| **Source regulatory documents** (247 documents) | **Per-source — see classification** | [`LICENSES/SOURCE_DOCUMENT_LICENSES.md`](LICENSES/SOURCE_DOCUMENT_LICENSES.md) |
| **Per-source attribution strings** | (required when using source content) | [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md) |

### Source document tier breakdown

A formal licensing audit (`LICENSES/SOURCE_DOCUMENT_LICENSES.md`) classified each of the 247 source documents into one of three tiers:

- **Tier 1 — Open licensing** (166 documents, 11 sources). Redistributed in full under each source's open licence (CC BY 4.0, Crown Copyright OGL, EU Reuse Decision 2011/833/EU, equivalents) with attribution. Sources: APRA, ASIC, AUSTRAC, ATO, AGSA, OAIC, RBA, Treasury (Commonwealth), ESMA, EU OJ, EBA-open.
- **Tier 2 — Academic-use redistribution** (26 documents, 4 sources). Redistributed under each source's academic-use terms with attribution; non-commercial bulk reuse only. Sources: EBA-conditional, SSO (BCBS papers), BIS-Basel, IOSCO.
- **Tier 3 — Metadata-only** (55 documents, 3 sources). The release includes title, jurisdiction, issuing body, source URL, and SHA-256 checksum — but **not** the document content. Users retrieve the originals from the source via `src/tools/fetch_tier3.py`. Sources: MAS (34), FATF (19), IRAS (2).

The audit's verbatim per-source license-clause quotations are preserved in [`LICENSES/SOURCE_DOCUMENT_LICENSES.md`](LICENSES/SOURCE_DOCUMENT_LICENSES.md).

---

## Contributing

We welcome contributions that expand the dataset's coverage or improve methodology. The most useful directions:

- **Annotations from jurisdictions outside the current six.** US, UK, Canada, ASEAN beyond Singapore.
- **Annotations in regulatory domains outside finance.** Healthcare, environmental, data protection outside finance, building codes, aviation.
- **Refinements to existing labels** following the published guidelines in `data/annotations/guidelines.md`, especially on cases where the published baselines showed low agreement.
- **Additional baseline implementations** — frontier commercial LLMs, legal-domain-pretrained encoders, retrieval-augmented systems, neuro-symbolic methods over regulatory relationship graphs.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) (forthcoming) for the contribution protocol. Contributions are accepted under the same layered licensing scheme as the existing release.

---

## Maintenance and versioning

RegConflict follows [semantic versioning](https://semver.org/). The release coinciding with the EMNLP 2027 paper is **v1.0.0**; subsequent point releases (v1.0.1, v1.1.0, …) carry corrections, expansions, and methodology improvements within the same data schema. Breaking changes (schema revision, typology revision) bump the major version.

| Release | Status |
|---|---|
| **v0.1.0** | Repository skeleton (this state). |
| **v1.0.0** | Coincides with EMNLP 2027 paper publication. Tagged release; Zenodo deposit; HuggingFace mirror. |
| **v1.x.y** | Corrections, additional baselines, expanded IAA studies. Same schema. |
| **v2.0.0** | Major dataset expansion (additional jurisdictions or domains). Schema revision possible. |

Each tagged release receives its own Zenodo deposit with a unique DOI; the conceptual DOI in `CITATION.cff` always resolves to the latest version.

---

## Contact

Repository maintainers: anonymised for double-blind review. Post-acceptance contact details land at v1.0 release.

For substantive issues during the submission review period, please file a GitHub issue (anonymously if needed) rather than emailing the maintainers directly.

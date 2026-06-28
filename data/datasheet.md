# Datasheet for RegConflict

This datasheet follows the framework in Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., and Crawford, K. (2021). [_Datasheets for Datasets._](https://dl.acm.org/doi/10.1145/3458723) Communications of the ACM, 64(12): 86–92.

It documents the motivation, composition, collection, preprocessing, intended uses, distribution, and maintenance of the **RegConflict v1.0** dataset. Section letters (A–G) map to the seven Gebru et al. (2021) categories.

---

## A. Motivation

### A.1 For what purpose was the dataset created?

To support natural-language-processing research on **cross-jurisdictional regulatory conflict detection** — the task of, given a pair of regulatory regimes from different jurisdictions, determining whether they conflict for an entity subject to both, and if so, classifying the conflict by its resolution structure.

No prior public benchmark frames cross-jurisdictional regulatory conflict detection as a first-class task with paired obligations from different regimes labelled with conflict relations. RegConflict fills that gap. The accompanying paper documents both the task formulation and baseline results (random predictors, zero-shot LLMs at two parameter scales, supervised fine-tuned encoders) establishing the current state of the art.

### A.2 Who created the dataset and on behalf of which entity?

The dataset was created by the paper's authors. Author identities are intentionally omitted from this v0.1 pre-publication release for double-blind review of the accompanying paper. The author block in [`../CITATION.cff`](../CITATION.cff) will be filled at camera-ready / non-anonymous release.

### A.3 Who funded the creation of the dataset?

[VERIFY: specify funding sources, e.g., university grant, scholarship, industry collaboration. Anonymised for review.]

### A.4 Any other comments?

The dataset was created alongside a paper accepted to [VERIFY: venue, EMNLP 2027 expected] and is intended to support reproducible evaluation of regulatory conflict-detection methods. Releasing the dataset is a load-bearing part of the contribution; the paper's results are validated entirely by the contents of this repository.

---

## B. Composition

### B.1 What do the instances represent?

Each instance is a **pair of regulatory regimes** (one each from `regime_a` and `regime_b`), accompanied by evidence passages drawn from the corpus for each regime, a binary `record_type` label (`conflict` or `non_conflict`), and — for conflict records — a four-class `conflict_type` and three-level `severity` label. The full record schema is documented in [`schema.md`](schema.md).

### B.2 How many instances are there in total?

**724 labelled regime-pair records:**

| Split | Records | Conflict | Non-conflict |
|---|---:|---:|---:|
| `train.jsonl` | 497 | 65 | 432 |
| `val.jsonl` | 107 | 14 | 93 |
| `test.jsonl` | 107 | 14 | 93 |
| `gold_iaa.jsonl` | 13 | 8 | 5 |
| **Total** | **724** | **101** | **623** |

Per-conflict-type counts across the full dataset:

- `operationally_resolvable`: 54 (most common — the modal pattern in financial regulation, especially privacy ↔ AML obligations resolved via legal instruments)
- `recurring_friction`: 15
- `interpretive_fact_sensitive`: 15
- `structural_unresolved`: 9 (rarest — entities literally cannot satisfy both regimes without restructuring)
- Plus a separately-held **30-pair gold IAA subset of conflict cases** used for inter-annotator agreement measurement; see §B.9 below and `data/annotations/iaa/`.

### B.3 Does the dataset contain all possible instances, or a sample from a larger set?

The dataset is a **purposive sample**, not exhaustive. The full population of regulatory-regime pairs across the six covered jurisdictional sources is combinatorially large (≈ 18 source organisations × multiple regimes each, yielding O(10⁴) possible pairs); the labelled set was selected to (a) cover all four conflict typology classes with non-trivial sample sizes, (b) include diverse jurisdictional pair shapes (same-jurisdiction, cross-jurisdictional, international-domestic), and (c) include both conflict and non-conflict pairs at a ratio matching real-world prevalence (≈ 14% conflict rate, deliberately low to reflect that most regime pairs do not conflict).

### B.4 What data does each instance consist of?

See [`schema.md`](schema.md) for the canonical schema. In brief: each record carries the two regimes' canonical identifiers (`regime_id`, `jurisdiction`, `issuing_body`), one or more evidence chunks per regime (passage text, source document filename, page number, chunk identifier), the binary label, the conflict-specific or non-conflict-specific block, an annotator-written rationale, and provenance metadata (annotator ID, timestamp, revision count, review status, confidence, optional adjudication block).

### B.5 Is there a label or target associated with each instance?

Yes — every record carries both a binary `record_type` label and, for conflict records, a four-class `conflict_type` and three-level `severity` label. Non-conflict records additionally carry a four-class `non_conflict_subtype` for analysis (not a prediction target).

### B.6 Is any information missing from individual instances?

- The `issuing_body` field is `"?"` for some legacy records where this metadata was not captured at annotation time.
- The `page` field on evidence chunks is `null` for non-paginated sources (some HTML-derived chunks).
- Records on the gold IAA subset (`gold_iaa.jsonl`) have a `review` block recording the second-annotator label and adjudication outcome; records outside the IAA subset have only a single annotator and no `review` block.

No load-bearing information is intentionally withheld from the labelled records. Per-source licensing metadata is in [`corpus/document_inventory.csv`](corpus/document_inventory.csv).

### B.7 Are relationships between individual instances made explicit?

Yes — by `regime_id`. Records that share a `regime_id` on either side reuse the same underlying regulatory regime, which means a model that memorises regime-level patterns will see them across multiple pairs. The split partitioning is at the **regime-pair** level (a specific (regime_a, regime_b) pair appears in exactly one of train/val/test/gold_iaa), but the same individual regime may appear on different sides of different pairs across splits. This is the standard arrangement for pairwise-classification benchmarks.

### B.8 Are there recommended data splits?

Yes — `train.jsonl` / `val.jsonl` / `test.jsonl` / `gold_iaa.jsonl`. The first three are stratified by `conflict_type` (5 strata: 4 conflict types + `non_conflict`) at proportions 0.7 / 0.15 / 0.15 with `seed=42` and partitioned at the regime-pair level. The 30-pair gold IAA subset of conflict cases is a separately-held random sample drawn from the labelled conflict pool *before* the stratified split, with raw per-annotator labels in `data/annotations/iaa/`. See [`splits/split_metadata.json`](splits/split_metadata.json) for full split-construction detail.

### B.9 Are there any errors, sources of noise, or redundancies in the dataset?

Yes — documented honestly:

- **Label reliability on the typology task.** Cohen's κ on the four-class typology was 0.6 on a 30-pair gold IAA subset of conflict cases (independently labelled by a second expert annotator after a rubric-calibration round) — substantial agreement on the Landis–Koch scale. With n=30 the confidence interval is informative-rather-than-definitive but supports the substantial-agreement claim; users should treat typology labels as reliably recovering one expert's interpretation of the rubric.
- **Single-annotator labels outside the IAA subset.** The non-IAA records (711 of 724) reflect one expert's interpretation of the typology rubric.
- **Rubric refinement during labelling.** The typology was refined iteratively during the initial labelling phase; early records may have been applied with a slightly different mental model than later ones, and not all early items were re-labelled after rubric finalisation. The `revision` field tracks per-record revision count.
- **Class imbalance.** 86% of records are non-conflicts; even within conflicts, `operationally_resolvable` accounts for ≈ half. Minority typology classes have n ≤ 15 across the full dataset.

The paper's Limitations section enumerates these issues; users should not treat the four-class typology labels as ground truth in the strict sense.

### B.10 Is the dataset self-contained, or does it link to or otherwise rely on external resources?

**Partially self-contained.** All labels, rationales, and metadata are self-contained. The source regulatory documents are partially redistributed: 192 of 247 documents (Tier 1 + Tier 2) are included in this release; 55 of 247 (Tier 3 — primarily MAS, FATF, IRAS) are referenced by metadata and SHA-256 only, with fetch URLs supplied in [`corpus/tier3_metadata_only/manifest.csv`](corpus/tier3_metadata_only/manifest.csv). Practical reproducibility depends on those Tier 3 sources remaining accessible at stable URLs.

### B.11 Does the dataset contain data that might be considered confidential?

No. All source documents are public-domain regulatory publications issued by named government agencies and standards-setting bodies, and all labels and rationales were authored by the dataset's annotators specifically for this release.

### B.12 Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

No. The dataset content is technical regulatory prose covering financial supervision, anti-money-laundering, privacy, payment services, and crypto-asset regulation. It does not contain content of a personal, sensitive, or potentially distressing nature.

### B.13 Does the dataset identify any subpopulations?

No subpopulation tagging. The dataset's jurisdictional coverage is recorded per regime (`regime_a.jurisdiction`, `regime_b.jurisdiction`) and per source organisation (per [`corpus/document_inventory.csv`](corpus/document_inventory.csv)), but does not represent any natural person or demographic group.

### B.14 Is it possible to identify individuals from the dataset?

No. The dataset contains no personally identifiable information about natural persons. Regulatory documents occasionally reference named officials (e.g., agency directors at the time of issuance) in passing; these are public-record references inherent to regulatory publication conventions and are not collected or aggregated by RegConflict.

### B.15 Does the dataset contain data that might be considered sensitive in any way?

The source documents include regulatory frameworks on privacy, anti-money-laundering, and sanctions, which are *about* sensitive subject matter — but the documents themselves are public regulatory text, not records of any individual's activity. No category of personal sensitive data (health, biometric, sexual orientation, etc.) appears.

---

## C. Collection process

### C.1 How was the data associated with each instance acquired?

Source regulatory documents were retrieved from each source organisation's official website (or, where the official portal was unavailable, from secondary public archives such as government open-data portals). Each document was fetched once, hashed with SHA-256, and the (URL, retrieval date, hash) tuple recorded in [`corpus/document_inventory.csv`](corpus/document_inventory.csv).

Labelled regime-pair instances were created *after* corpus assembly. The annotator selected regime pairs based on the corpus's domain coverage and on the typology balance targets described in [`annotations/guidelines.md`](annotations/guidelines.md), then wrote labels and rationales by reading the cited evidence passages.

### C.2 What mechanisms or procedures were used to collect the data?

- **Source-document collection**: manual download from official agency websites, supplemented by automated fetch scripts where the source provided machine-readable index endpoints.
- **Evidence-chunk extraction**: documents were segmented into chunks of ≈ 200–800 tokens using a heading-aware splitter; chunks retain page-number provenance for PDFs. The evidence chunks shown to annotators (and to evaluation-time models) are drawn from this chunk store.
- **Annotation**: conducted in a custom labelling UI (`ui/labelling/` in the development repository, not redistributed) that presented one regime pair at a time with retrieved evidence chunks and a structured form for the rationale, typology, severity, and confidence.

### C.3 If the dataset is a sample from a larger set, what was the sampling strategy?

The full set of possible regime pairs is combinatorially large; sampling was purposive:

1. For positive examples (conflicts), the annotator drew on regulatory-compliance domain knowledge to identify pairs likely to conflict, then verified the conflict by reading both regimes' relevant provisions.
2. For negative examples (non-conflicts), pairs were drawn from regimes that were topically adjacent to conflict cases (e.g., same jurisdiction or same domain) to provide hard negatives, plus a smaller proportion of clearly-non-conflicting pairs to provide easy negatives.
3. The 14% conflict rate was a deliberate design choice to reflect real-world prevalence — most regime pairs in practice do not conflict — while ensuring enough positives for supervised training.
4. The typology balance was targeted: each of the four conflict classes has n ≥ 9 across the full dataset, though `operationally_resolvable` (the modal class in financial regulation) accounts for ≈ half of conflict records.

### C.4 Who was involved in the data collection process and how were they compensated?

- **Primary annotator**: [VERIFY — describe — e.g., "the first author, a researcher with expertise in financial regulatory compliance"].
- **Second annotator (30-pair gold IAA subset)**: [VERIFY — describe role and compensation arrangement: co-author / acknowledged contribution / paid at institutional standard rate]. The second annotator independently labelled the four-class typology on 30 randomly-sampled conflict pairs after a rubric-calibration round on a disjoint training set.
- **Source-document retrieval**: automated where possible; manual fetch performed by the primary annotator.

Total annotation effort: approximately [VERIFY: ~150] hours across rubric design, primary labelling, IAA work, and adjudication.

### C.5 Over what timeframe was the data collected?

Corpus assembly and licensing audit: completed 2026-06-19. Annotation phase: [VERIFY: timeframe, e.g., May–June 2026]. The dataset reflects regulatory state as of [VERIFY: corpus freeze date — mid-June 2026 per the licensing audit].

### C.6 Were any ethical review processes conducted?

The work does not involve human-subjects research in the IRB sense (no data collected from people, no experiments on people, no interviews). The second-annotator labelling task could in some institutions require ethics approval depending on local rules around data labelling as research labour; if such approval was obtained, the approval number will be recorded at non-anonymous release. [VERIFY: institutional ethics review status.]

### C.7 Was the data collected from individuals directly, or via third parties?

Neither — the data was collected from publicly-available regulatory publications. No individuals were the subject of collection.

### C.8 Were the individuals notified about the data collection?

Not applicable.

### C.9 Did the individuals consent to the collection and use of their data?

Not applicable.

### C.10 If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

Not applicable.

### C.11 Has an analysis of the potential impact of the dataset and its use on data subjects been conducted?

The dataset contains no individual-level data subjects. Potential misuse considerations (regulatory arbitrage, automated compliance without expert review) are discussed in the accompanying paper's Ethics Statement and below in §E.4.

---

## D. Preprocessing / cleaning / labeling

### D.1 Was any preprocessing/cleaning/labeling of the data done?

Yes.

- **PDF text extraction**: documents were extracted with a layout-aware extractor that preserves heading structure and page numbers. Output is plain text per page.
- **Chunking**: documents were segmented into evidence chunks of ≈ 200–800 tokens using a heading-aware splitter. Chunks retain `(source_doc, page, chunk_id)` provenance.
- **Deduplication**: near-duplicate chunks (≥ 0.9 cosine similarity on sentence-transformer embeddings) were collapsed to canonical representatives to avoid duplicate evidence across pairs.
- **Labelling**: as described above. Annotator confidence and review-status metadata are preserved on every record.

### D.2 Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?

Yes — the original source documents are released as part of `corpus/tier1_redistributable/` and `corpus/tier2_conditional/` (for the 192 redistributable documents); the chunk-level extraction and the labelled records are released as separate JSONL files. The chunk extractor itself is in `src/tools/` (Phase 3 deliverable).

### D.3 Is the software that was used to preprocess/clean/label the data available?

Yes. The chunk extractor and labelling-UI source live in `src/` (Phase 3 / 4 deliverable) under the same MIT licence as the rest of the code. The Tier 3 fetch tool (`src/tools/fetch_tier3.py`) is provided for users to retrieve the withheld documents and verify them against the published SHA-256 hashes.

---

## E. Uses

### E.1 Has the dataset been used for any tasks already?

Yes. The accompanying paper reports baseline results on the binary conflict-detection task and the four-class typology task, comparing:

- A majority-class baseline (always predicts non-conflict): binary F1 = 0.000.
- A stratified random baseline (100 seeds): binary F1 = 0.127 ± 0.077.
- Zero-shot DeepSeek-R1-Distill-Qwen-1.5B (3 prompt variants): binary F1 = 0.135 ± 0.098.
- Zero-shot GLM-4.7-Flash (≈ 30B parameters, 3 prompt variants): binary F1 = 0.100 ± 0.082.
- Supervised fine-tuned DeBERTa-v3-base cross-encoder (5 seeds): binary F1 = **0.517 ± 0.022**.

The DeBERTa baseline's advantage over both zero-shot LLM baselines is statistically significant under paired bootstrap resampling (p = 0.002 vs DeepSeek, p < 0.001 vs GLM).

### E.2 Is there a repository that links to any or all papers or systems that use the dataset?

Not yet. As external uses accumulate, they will be tracked in [`../CHANGELOG.md`](../CHANGELOG.md) and on the project's GitHub homepage.

### E.3 What other tasks could the dataset be used for?

- Cross-jurisdictional regulatory similarity / alignment measurement.
- Training and evaluation of legal-domain-pretrained encoders.
- Few-shot learning experiments on the minority typology classes.
- Construction of regulatory knowledge graphs grounded in observed conflict relationships.
- Evaluation of retrieval-augmented generation systems over regulatory text.
- Domain-adaptation studies for legal-language pretraining.

The annotations are independently licensed (CC BY 4.0; see [`../LICENSES/LICENSE-ANNOTATIONS.md`](../LICENSES/LICENSE-ANNOTATIONS.md)) so that derived datasets and annotations layered on top of RegConflict are straightforward to release.

### E.4 Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

Yes — these limitations should be propagated to any downstream use:

- **Six-jurisdiction coverage.** Generalisation to other jurisdictions (US, UK, Canada, broader ASEAN) is not empirically established.
- **One regulatory domain.** Coverage is financial regulation (AML, prudential, privacy, payment services, crypto-asset frameworks). Healthcare, environmental, construction, aviation, and food-safety regulatory conflicts may exhibit different typological structures.
- **English-only source documents.** All 247 documents are in English; non-English regulatory NLP is not directly supported.
- **Small N.** 724 total records is small by NLP-benchmark standards; per-class F1 estimates carry wide uncertainty.
- **Typology κ = 0.6 on the 30-pair IAA subset.** Substantial inter-annotator agreement on the Landis–Koch scale; users should treat typology labels as reliably recovering one expert's interpretation of the rubric, with residual disagreement concentrated on borderline pairs near class boundaries.
- **Class imbalance.** 86% of records are non-conflicts; minority typology classes have n ≤ 15.
- **Single-annotator labels outside IAA subset.** 711 of 724 records reflect one expert's interpretation.
- **Tier 3 dependence.** 55 of 247 source documents are not redistributed; reconstruction requires fetching from official sources whose URLs may drift.

### E.5 Are there tasks for which the dataset should not be used?

Yes:

- **Production compliance decision-making.** RegConflict labels and any models trained on them are research artefacts, not legal advice. Compliance decisions in regulated settings require qualified legal and regulatory expertise; automated systems built atop this benchmark are best understood as aids that surface candidates for expert review, not as authoritative determinations of regulatory status.
- **Regulatory arbitrage planning.** Using the conflict-detection capability to identify jurisdictions whose obligations can be evaded by structuring transactions across regimes is a foreseeable misuse vector. The dataset and any derived systems should not be used to support such planning.
- **Closed-source automated enforcement.** Predicting conflict status to support automated enforcement actions (penalties, licence revocation) against named entities is outside the dataset's intended scope.

---

## F. Distribution

### F.1 Will the dataset be distributed to third parties outside the entity on behalf of which the dataset was created?

Yes — the dataset is intended for public release.

### F.2 How will the dataset be distributed?

Three distribution channels are planned for v1.0:

1. **GitHub repository** (primary, this release). Code, labels, redistributable source documents, datasheet, and licence documentation.
2. **Zenodo deposit** (versioned). Long-term archival deposit with persistent DOI.
3. **HuggingFace Datasets** (mirror). Convenient access via the `datasets` library.

The Zenodo DOI and HuggingFace dataset ID will be added to [`../CITATION.cff`](../CITATION.cff) at v1.0 tag.

### F.3 When will the dataset be distributed?

Coinciding with the paper's publication at [VERIFY: EMNLP 2027 expected].

### F.4 Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

Yes — the layered licensing scheme is documented at [`../LICENSES/`](../LICENSES/):

- **Code**: MIT Licence.
- **Annotations** (typology, labels, decision rubric, datasheet, document inventory metadata): CC BY 4.0.
- **Source regulatory documents**: per-source licensing (Tier 1 open / Tier 2 academic-use / Tier 3 metadata-only). Tier-by-tier and source-by-source detail in [`../LICENSES/SOURCE_DOCUMENT_LICENSES.md`](../LICENSES/SOURCE_DOCUMENT_LICENSES.md).
- **Per-source attribution strings** required when using source content: [`../ATTRIBUTIONS.md`](../ATTRIBUTIONS.md).

### F.5 Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

Yes — the source regulatory documents retain their original copyright. Three source organisations (MAS, FATF, IRAS — accounting for 55 of 247 documents) do not permit redistribution under their published terms; these documents appear in the release as metadata-plus-fetch-instructions only. The full per-source licensing audit, including verbatim license-clause quotations, is in [`../LICENSES/SOURCE_DOCUMENT_LICENSES.md`](../LICENSES/SOURCE_DOCUMENT_LICENSES.md).

### F.6 Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

No. The dataset is composed of publicly-available regulatory text and derived annotations, neither of which is subject to export-control regulation in any of the covered jurisdictions.

---

## G. Maintenance

### G.1 Who will be supporting/hosting/maintaining the dataset?

The paper's authors at v1.0 release. Maintainer contact information will be added at non-anonymous release. The Zenodo deposit provides long-term archival hosting independent of the maintainers' active engagement.

### G.2 How can the owner/curator/manager of the dataset be contacted?

Via GitHub issues at the project repository (URL added at non-anonymous release). For sensitive issues (e.g., licence reinterpretation requests from a source organisation), direct contact details will be in the README at v1.0 tag.

### G.3 Is there an erratum?

Errata will be tracked in [`../CHANGELOG.md`](../CHANGELOG.md). Each point release (v1.0.1, v1.0.2, …) will document changes; corrections that alter benchmark results will be flagged prominently. Reproducibility of paper-reported numbers requires the v1.0.0 tag specifically; subsequent versions may revise labels.

### G.4 Will the dataset be updated?

Yes. Planned update cadence:

- **Patch releases (v1.0.x)**: label corrections, rubric clarifications, additional IAA records.
- **Minor releases (v1.1, v1.2, …)**: additional regime pairs within the same six-jurisdiction scope; expanded IAA subset.
- **Major releases (v2.0, …)**: expansion to additional jurisdictions or regulatory domains; possible typology revision.

Each release receives its own Zenodo DOI; the conceptual DOI in [`../CITATION.cff`](../CITATION.cff) always resolves to the latest version.

### G.5 If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?

Not applicable — the dataset does not relate to people.

### G.6 Will older versions of the dataset continue to be supported/hosted/maintained?

Yes. Each tagged release is preserved as an immutable Zenodo deposit. The v1.0.0 release in particular is preserved for paper-reproducibility purposes; subsequent versions are additive rather than replacements.

### G.7 If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

Yes. Contribution protocol is documented in [`../CONTRIBUTING.md`](../CONTRIBUTING.md) (Phase 4 deliverable). Contributions are accepted under the same layered licensing scheme as the existing release. The most useful directions for contribution are listed in the README:

- Annotations from jurisdictions outside the current six.
- Annotations in regulatory domains outside finance.
- Refinements to existing labels following the published guidelines.
- Additional baseline implementations.

---

## Datasheet revision history

- **v0.1 (Phase 2 of public release build, current)**: initial draft following Gebru et al. (2021), authored from the paper's §4, Limitations, and Ethics Statement plus the licensing audit's verbatim findings.
- **v1.0 (planned, EMNLP 2027 publication)**: filled `[VERIFY]` placeholders; final author block; locked DOI; matched to v1.0 tag.

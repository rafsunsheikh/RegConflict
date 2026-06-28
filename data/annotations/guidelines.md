# Annotation guidelines

These are the operational guidelines used to label the RegConflict v1.0 dataset. They are reproduced here so that (a) downstream users understand how the labels were produced, and (b) contributors to v1.1+ releases can apply consistent labels.

This document complements [`decision_rubric.md`](decision_rubric.md), which defines the four-class typology in detail. The rubric is *what* to label. This document is *how* to label, including how to select pairs, what to read, how to write the rationale, and how to handle uncertainty.

## Before you start

- **Read the rubric in full.** [`decision_rubric.md`](decision_rubric.md). The rubric defines the four classes operationally; familiarity with the operational definitions (not just the names) is essential.
- **Read 5–10 already-labelled records.** Sample randomly from `data/conflicts/train.jsonl`. Read the regime pair, the rationale, and the assigned label. The labels define the typology by example as much as the rubric does.
- **Identify your background.** If you have regulatory or compliance expertise in a specific domain (e.g., AML, privacy, payments), bias your contributions toward pairs in that domain — your labels will be more reliable there.

## What to label

The v1.0 dataset was built by labelling pairs in this order. Future contributors should follow the same sequence:

### Phase A — High-confidence positive examples (~15 per contributor)

Label the most uncontroversial conflicts first. These give the model strong positive examples and establish your interpretation of the typology for the rest of your contribution.

Aim for at least 3 of each conflict type, drawn from regimes you have direct expertise in. For each, write the full record: regime pair, evidence passages, rationale, conflict type, severity.

### Phase B — High-confidence negative examples (~15)

Label pairs that share context but don't conflict. These are the hard negatives the model needs.

Examples of useful hard-negative shapes:

- **Same jurisdiction, both apply to most regulated activity, but no obligation tension.** E.g., AU AML/CTF ↔ AU Privacy as a baseline (the conflict is between AML *travel rule* and AU_PRIVACY *APP8* specifically; the regimes broadly coexist).
- **Different jurisdictions, both regulating crypto-asset activity, but operationally non-overlapping.** E.g., EU MiCA Title II (other crypto-assets) ↔ AUDD (which is EMT-like, Title IV) — no conflict because Title II doesn't apply.
- **Aligned implementation pairs.** E.g., FATF Recommendation 16 ↔ AUSTRAC travel rule — these are aligned, not in conflict.

For non-conflict records, pick the `non_conflict_subtype` from `{aligned_complementary, different_domains, superficially_similar_but_disjoint, same_jurisdiction_no_overlap}`. This taxonomy is for analysis, not part of the prediction task — but be consistent.

### Phase C — Borderline cases (~10)

The most valuable. Pairs where you had to think hard about the typology. For each borderline case, the *rationale* matters more than the label.

Examples of borderline shapes:

- "Are these in conflict, or do they just apply in different layers?" — Often `interpretive_fact_sensitive` or non-conflict / `aligned_complementary`.
- "Neither blocks the other, but a firm operating in both has to navigate both" — Often `recurring_friction` or non-conflict / `aligned_complementary`.
- "Privacy says delete, AML says keep" — Almost always `operationally_resolvable` (via lawful basis + retention exemption), but the resolution is sometimes fact-dependent.

For these cases, write the rationale as if it were a paragraph in the paper. Other annotators will read it to calibrate their own labels.

## What to read for each pair

For each regime, retrieve passages that contain:

1. The **scope** clause(s) of the regime — what entities/activities it applies to.
2. The **specific obligations** that bear on the conflict — the actual conduct required or prohibited.
3. Any **exemptions, equivalence determinations, or carve-outs** that affect the conflict's resolution.

Three to five chunks per side is the typical range. Less than two is rarely enough; more than seven risks burying the load-bearing passages in noise. The evaluation harness shows models the chunks you select, so the selection is part of the annotation.

Use the corpus search tools — `src/tools/search_corpus.py` (Phase 3) provides regex and semantic search over the indexed corpus. For Tier 3 documents that are metadata-only, fetch them via `src/tools/fetch_tier3.py` before annotation.

## How to write the rationale

A good rationale answers four questions:

1. **What does each regime require?** One sentence per side, naming the specific provision.
2. **Where does the tension arise?** Name the specific obligation pair that produces the conflict.
3. **Why does it fall in the chosen typology class?** Tie back to the decision rubric — name the resolution mechanism (or absence) that places this case in the chosen class.
4. **What's the severity reasoning?** Why low/medium/high for an entity subject to both.

Length: aim for 60–150 words. Shorter than 60 words is often too thin to support the label; longer than 150 starts to repeat the regulatory text rather than the conflict analysis.

Rationale style:

- **Cite passages by section number where possible** (e.g., "MiCA Art. 81(15)(a)") rather than by chunk position.
- **Avoid hedging language** ("might possibly perhaps conflict") in the final rationale. If you're uncertain, set `confidence` lower and use `claude_caveats` for the uncertainty, not the rationale.
- **State the entity perspective explicitly** — "for an entity subject to both X and Y" — rather than leaving it implicit. A conflict is always between obligations on a *party*.

## Confidence and caveats

- **`confidence` ∈ [0, 1]**: Your self-assessed probability that another reasonable annotator would assign the same binary label and the same typology class.
  - 0.9–1.0: clear, well-precedented conflict; another expert almost certainly agrees.
  - 0.7–0.9: confident but with a plausible alternative interpretation; rationale should note it.
  - 0.5–0.7: genuinely borderline; expect typology disagreement.
  - Below 0.5: don't label as confidently classed; either downgrade to non-conflict or flag for adjudication.
- **`claude_caveats`** (array of strings): note edge cases, alternative interpretations, the second-best class, or specific facts that would flip the label. Empty `[]` is fine when there are none.
- **`review_status`**: leave as `"single_annotation"` unless your contribution flow includes a second annotator review pass.

## Handling uncertainty

When you're not sure:

- **Between conflict and non-conflict**: prefer `non_conflict` unless you can name a specific obligation pair that creates the tension. Topical proximity is not sufficient.
- **Between typology classes**: apply the tiebreaker rule in [`decision_rubric.md`](decision_rubric.md) (structural > interpretive > operational > recurring), and note the second-best class in the rationale.
- **Between conflict severities**: prefer `medium` for most legitimate conflicts. Reserve `high` for conflicts that affect core business viability and `low` for marginal operational impact.

When you genuinely don't know:

- Set `confidence` low, write up the case in `claude_caveats`, set `human_review_required: true`, and move on. Forced confident labelling on uncertain cases poisons the dataset.

## Inter-annotator agreement protocol

The v1.0 release includes a **30-pair gold IAA subset of conflict cases** labelled independently by two expert annotators. The IAA protocol:

1. Annotator 1 (primary) labels the full pool.
2. A 30-pair IAA sample is drawn by random sampling from the labelled conflict pool (`record_type == "conflict"`) with `seed=42`. Sampling is conditioned on positive primary-annotator labels to focus the IAA exercise on the four-class typology — the task where reliability evidence is most informative.
3. The primary and second annotators jointly walk through the decision rubric on a small set of training pairs *not* drawn from the IAA sample (rubric-calibration round). This establishes a shared operational understanding of the four-class typology, tiebreaker rule, and severity scale.
4. Annotator 2 then labels the 30 IAA pairs blind — without seeing Annotator 1's labels, rationales, or rationales for any pair in the IAA sample. Only the regime pair identifiers and cited evidence passages are visible.
5. Disagreements (typology) are adjudicated in a joint review session; both annotators discuss until consensus; the consensus label is recorded as the gold label, with both annotators' original blind labels preserved in `annotations/iaa/annotator1_labels.jsonl` and `annotations/iaa/annotator2_labels.jsonl`.

Reported IAA on the v1.0 30-pair gold subset:

- Cohen's κ (4-class typology): **0.6** — substantial agreement on the Landis–Koch (1977) scale.

Binary IAA is not separately reported because the sample is conditioned on positive primary labels (all 30 pairs are confirmed conflicts), so there is no binary variance to compute κ over. Binary IAA on the full label distribution should be the target of future IAA studies that draw from the full labelled pool.

The typology κ should be read as a finding that the four-class schema is recoverable by a second expert annotator to a substantial-agreement level after rubric calibration, with the residual disagreement concentrated on borderline pairs near class boundaries. The κ computation is reproducible via `scripts/compute_iaa.py`.

## Contribution workflow

If you are contributing labels to a future release (v1.1+):

1. **Open a GitHub issue** describing the regimes you plan to label (jurisdictions, regime IDs, expected count, your domain background).
2. **Wait for confirmation** from the maintainers that the proposed scope doesn't conflict with another in-progress contribution.
3. **Label using this guidelines document + the decision rubric**. Use the schema in [`../schema.md`](../schema.md).
4. **Run the verify tool** on your contribution: `python src/tools/verify_corpus.py --records your_contribution.jsonl` — this catches schema errors, missing evidence references, and class-distribution skew before submission.
5. **Submit a pull request.** Include the records, the IAA subset if you produced one, and a contribution summary noting any rubric edge cases you encountered.

See [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) for the full PR template and review process. (Phase 4 deliverable; not yet present.)

## What changed in v1.0 labelling that downstream annotators should know

The v1.0 labelling process was iterative:

- The typology was initially defined as a 3-class schema and expanded to 4 classes mid-labelling, when `recurring_friction` was identified as distinct from `operationally_resolvable`.
- Some early records (Phase A from the original session) were re-labelled after the rubric was finalised; others were not. The `revision` field tracks per-record revision count, and `review_status` tracks adjudication status.
- The class-distribution targets were rebalanced during labelling: the initial pass over-represented `operationally_resolvable` (the most common shape in financial regulation); a second pass targeted the three minority classes specifically. The final v1.0 distribution still skews toward `operationally_resolvable` (38 / 65 training conflicts), but the minority classes are present in n ≥ 6 each.

For v1.1+ annotators, the rubric is the authoritative source. The iterative-refinement history is preserved in the paper's Limitations section and in this dataset's CHANGELOG.

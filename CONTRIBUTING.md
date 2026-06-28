# Contributing to RegConflict

Contributions that expand the dataset's coverage, improve the methodology, or refine the rubric are very welcome. This document describes the contribution protocol.

The most useful contribution directions:

- **Annotations from jurisdictions outside the current six** (US, UK, Canada, ASEAN beyond Singapore).
- **Annotations in regulatory domains outside finance** (healthcare, environmental, data protection outside finance, building codes, aviation).
- **Refinements to existing labels** following the published guidelines, especially on cases where the published baselines showed low agreement (see `results/iaa/disagreement_log.jsonl` once populated).
- **Additional baseline implementations** — frontier commercial LLMs, legal-domain-pretrained encoders, retrieval-augmented systems, neuro-symbolic methods.
- **Expanded IAA studies** with k ≥ 3 second annotators and Fleiss' κ.
- **Code improvements** — performance, test coverage, documentation, type annotations, schema strictness.

## Licensing

All contributions are accepted under the layered licensing scheme described in [`README.md`](README.md) and [`LICENSES/`](LICENSES/):

- **Code** is contributed under the MIT Licence.
- **Annotations, decision-rubric refinements, and datasheet additions** are contributed under CC BY 4.0.
- **Source documents** retain their original licensing; if you contribute new source documents, you must classify them into Tier 1 / Tier 2 / Tier 3 per the audit framework in [`LICENSES/SOURCE_DOCUMENT_LICENSES.md`](LICENSES/SOURCE_DOCUMENT_LICENSES.md) and add per-source attribution strings to [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md).

By opening a pull request you confirm that you hold the rights to contribute the material under these terms.

## Annotation contributions

If you are contributing labels (new pairs or revisions to existing ones), follow this protocol:

1. **Open a GitHub issue** describing the regimes you plan to label (jurisdictions, regime IDs, expected count, your domain background). Wait for confirmation from maintainers that the proposed scope doesn't conflict with another in-progress contribution.
2. **Read the rubric and guidelines in full**:
   - [`data/annotations/decision_rubric.md`](data/annotations/decision_rubric.md) — operational definitions of the four-class typology.
   - [`data/annotations/guidelines.md`](data/annotations/guidelines.md) — labelling protocol, rationale writing, IAA expectations.
3. **Label using the canonical schema** documented in [`data/schema.md`](data/schema.md).
4. **Run the verify tool** on your contribution:

   ```bash
   python src/tools/verify_corpus.py --records your_contribution.jsonl
   ```

   This catches schema errors, missing evidence references, and class-distribution skew before submission.
5. **Submit a pull request** with:
   - The new records as a JSONL file.
   - A contribution summary noting any rubric edge cases you encountered.
   - If you produced a second-annotator IAA pass on your records, the per-annotator label files following the `data/annotations/iaa/README.md` schema.

The maintainers will review for schema validity, rubric consistency, and integration impact before merging.

## Code contributions

Code contributions follow the standard GitHub flow:

1. Fork the repository.
2. Create a topic branch (`feat/...`, `fix/...`, `docs/...`).
3. Make your changes.
4. Add or update tests under `tests/`. New code should be covered by at least one test; bugfixes should add a regression test.
5. Run the full test suite locally:

   ```bash
   pip install -r requirements.txt
   pytest tests/ -v
   ```

6. Run the linter (CI will run this; running locally first avoids round-trips):

   ```bash
   pip install ruff
   ruff check src/ scripts/ tests/
   ruff format --check src/ scripts/ tests/
   ```

7. Submit a pull request describing the change, the motivation, and the test coverage.

CI runs `pytest`, `ruff`, and the release-integrity check on every PR.

## Baseline contributions

New baseline implementations should:

- Live under `src/baselines/<your_baseline_name>/`.
- Produce predictions in the JSONL format documented in `src/eval/benchmark/io.py`.
- Be runnable via a script in `scripts/run_<your_baseline>.py` that accepts standard CLI flags (`--split`, `--seed`, `--output-dir`).
- Include a config file under `configs/` documenting hyperparameters and the reasoning for non-default choices.
- Add a results-table entry to `results/results_table.md` when integrated.
- Be reproducible: same seed must produce identical predictions.

## Rubric refinement contributions

If you find an edge case the existing rubric doesn't handle cleanly:

1. Open a `data-quality` issue (template: `.github/ISSUE_TEMPLATE/data_quality.md`).
2. In your write-up, cite the specific regulatory passage(s) and the operational test from the rubric that fails to disambiguate the case.
3. Propose a rubric refinement (one or two sentences for the decision_rubric.md).
4. If accepted, the maintainers will release a new minor version (v1.x) with the rubric refinement and any consequent label revisions.

## Issue templates

- **Bug reports**: use `.github/ISSUE_TEMPLATE/bug_report.md`.
- **Data quality concerns**: use `.github/ISSUE_TEMPLATE/data_quality.md`.

## Code of Conduct

All participants in the project (issue reporters, contributors, maintainers, reviewers) are expected to follow [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Maintainers

Contact details will be added at v1.0 non-anonymous release.

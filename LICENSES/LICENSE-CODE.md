# Code License — MIT

All code in this repository (evaluation harness, baseline implementations, fetch and verification utilities, tests, configuration loaders, notebooks) is released under the MIT License.

The canonical license text appears at the top of the repository as [`/LICENSE`](../LICENSE).

## Scope

This MIT License covers, but is not limited to:

- Everything under `src/` (evaluation harness, baseline implementations, tools).
- Everything under `scripts/` (CLI entry points for reproducing paper results).
- Everything under `tests/` (unit and integration tests).
- Everything under `configs/` (YAML configuration files).
- Everything under `notebooks/` (Jupyter notebooks demonstrating reproduction).
- Everything under `.github/` (CI workflows and issue templates).
- Build, lint, and packaging configuration (e.g., `pyproject.toml`, `requirements.txt`).

## Out of scope

The MIT License does **not** apply to:

- Labels, annotations, the four-class typology, the decision rubric, the datasheet, or any derived annotations layered on the source documents. Those are released under **CC BY 4.0** — see [`LICENSE-ANNOTATIONS.md`](LICENSE-ANNOTATIONS.md).
- The source regulatory documents themselves. Each retains its original licensing as published by the issuing authority. See [`SOURCE_DOCUMENT_LICENSES.md`](SOURCE_DOCUMENT_LICENSES.md) for per-source detail and [`/ATTRIBUTIONS.md`](../ATTRIBUTIONS.md) for required attribution strings.

## Citation

If you build research artefacts on top of this code, please cite the accompanying paper using the BibTeX entry in [`/CITATION.cff`](../CITATION.cff). Citation is not a license condition (the MIT License does not require it), but it is the standard academic courtesy and helps surface downstream work to other researchers.

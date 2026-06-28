# `src/` — RegConflict source code

All code in this directory is released under the **MIT License** (see [`/LICENSE`](../LICENSE) and [`/LICENSES/LICENSE-CODE.md`](../LICENSES/LICENSE-CODE.md)).

The module layout mirrors the conceptual structure of the task: a typed schema for labelled records, an evaluation harness that all baselines plug into, three baseline implementations (random / zero-shot / fine-tuned), and release-side tools (fetch, build splits, verify).

## Layout

```
src/
├── README.md                          ← this file
├── eval/
│   ├── benchmark/                     ← evaluation harness
│   │   ├── schema.py                  ← Pydantic v2 record schema
│   │   ├── io.py                      ← split loading + prediction I/O
│   │   ├── metrics.py                 ← F1, macro-F1, typology macro-F1, joint EM
│   │   ├── bootstrap.py               ← bootstrap CIs + paired bootstrap
│   │   ├── runner.py                  ← end-to-end evaluation driver
│   │   └── validators.py              ← runtime schema validation
│   └── __init__.py
│
├── baselines/
│   ├── random/                        ← majority class + stratified random
│   │   ├── majority.py
│   │   └── stratified.py
│   ├── zero_shot/                     ← LLM-prompted baselines
│   │   ├── client.py                  ← httpx client for llama.cpp / OAI API
│   │   ├── prompts.py                 ← three prompt variants (A/B/C)
│   │   ├── parser.py                  ← JSON-output parsing with retry policy
│   │   └── runner.py                  ← per-cell evaluation driver
│   └── fine_tuned/                    ← DeBERTa-v3-base cross-encoder
│       ├── data.py                    ← joint-budget tokenisation + DataLoader
│       ├── model.py                   ← two-head cross-encoder
│       └── train.py                   ← training loop + early stopping
│
└── tools/
    ├── fetch_tier3.py                 ← download Tier 3 docs + SHA-256 verify
    ├── build_splits.py                ← reproduce splits from raw labels
    └── verify_corpus.py               ← integrity check on the release
```

## How the pieces connect

1. **Data flows in through `src/eval/benchmark/io.py`.** This module loads `data/conflicts/*.jsonl` with the schema in [`schema.py`](eval/benchmark/schema.py), exposing a uniform `load_split(name)` function used by every baseline.
2. **Each baseline produces predictions.** The three baseline implementations (`baselines/random/`, `baselines/zero_shot/`, `baselines/fine_tuned/`) each emit a `predictions.jsonl` file in the format documented in [`io.py`](eval/benchmark/io.py).
3. **`src/eval/benchmark/runner.py` scores predictions.** It computes the metrics defined in [`metrics.py`](eval/benchmark/metrics.py), generates bootstrap CIs via [`bootstrap.py`](eval/benchmark/bootstrap.py), and writes a per-run output bundle under `results/`.
4. **The release-side tools live in `src/tools/`.** These are user-facing utilities for reconstructing the corpus (`fetch_tier3.py`), reproducing the splits from raw labels (`build_splits.py`), and verifying the release on disk (`verify_corpus.py`).

## Reproducing the paper

End-to-end reproduction:

```bash
# 1. Fetch Tier 3 documents (or skip if you're only using the labelled splits)
python src/tools/fetch_tier3.py

# 2. Verify the release on disk
python src/tools/verify_corpus.py

# 3. Random baselines (majority class + 100-seed stratified random)
python scripts/run_random_baselines.py

# 4. Zero-shot LLM baselines (requires llama.cpp servers running on 8080/8081 —
#    see Appendix D of the paper for launch commands)
python scripts/run_zero_shot_baseline.py

# 5. Fine-tuned DeBERTa baseline (5 seeds, on Apple Silicon or CUDA)
python scripts/train_deberta_baseline.py --mode multi_seed

# 6. Paired bootstrap significance test
python scripts/paired_bootstrap_baselines.py

# 7. Aggregate metrics
python scripts/run_eval.py --baseline all --split test
python scripts/run_eval.py --baseline all --split gold_iaa
```

Outputs land under `results/<baseline>/<cell_or_seed>/<split>_<timestamp>/`.

## Working with the schema

Lightweight dict access:

```python
import json
from pathlib import Path

records = [json.loads(l) for l in Path("data/conflicts/test.jsonl").read_text().splitlines() if l.strip()]
conflicts = [r for r in records if r["record_type"] == "conflict"]
```

Typed access with validation:

```python
from src.eval.benchmark.schema import LabelledPair

for line in Path("data/conflicts/test.jsonl").read_text().splitlines():
    if line.strip():
        record = LabelledPair.model_validate_json(line)
        if record.record_type == "conflict":
            print(record.pair_id, record.conflict_type, record.severity)
```

The Pydantic schema raises on records missing required type-specific fields (e.g., a conflict record without `conflict_type` or `severity`), which `verify_corpus.py` uses to flag schema errors before they reach the baselines.

## Configuration

Per-baseline hyperparameters live in [`/configs/`](../configs/):

- [`configs/deberta_baseline.yaml`](../configs/deberta_baseline.yaml) — DeBERTa-v3-base training config.
- [`configs/zero_shot_prompts.yaml`](../configs/zero_shot_prompts.yaml) — model endpoints, prompt variants, inference settings.
- [`configs/evaluation.yaml`](../configs/evaluation.yaml) — metrics, bootstrap parameters, output layout.

Override fields via CLI flags on the script entry points; the YAMLs document the v1.0 defaults.

## Tests

Unit and integration tests for the modules in this directory live in [`/tests/`](../tests/) (Phase 4 deliverable). Run with `pytest`:

```bash
pip install -r requirements.txt
pytest tests/
```

## License

MIT — see [`/LICENSE`](../LICENSE) and [`/LICENSES/LICENSE-CODE.md`](../LICENSES/LICENSE-CODE.md). The annotations and source documents this code operates on are released under separate licences; see [`/LICENSES/LICENSE-ANNOTATIONS.md`](../LICENSES/LICENSE-ANNOTATIONS.md) and [`/LICENSES/SOURCE_DOCUMENT_LICENSES.md`](../LICENSES/SOURCE_DOCUMENT_LICENSES.md).

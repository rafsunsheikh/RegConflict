# `notebooks/` — RegConflict reproduction and exploration

Three notebooks for working with the released benchmark. They are authored as `.py` files in [jupytext "percent" format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html) so they:

- Run directly as Python scripts (`python notebooks/01_dataset_exploration.py`).
- Convert to Jupyter `.ipynb` notebooks with a single command (`jupytext --to ipynb notebooks/01_dataset_exploration.py`).
- Stay diffable in version control without binary cell-output noise.

## Notebooks

| File | Purpose | Runtime | Prerequisites |
|---|---|---|---|
| [`01_dataset_exploration.py`](01_dataset_exploration.py) | Per-split counts, per-class distributions, jurisdictional pair breakdown, sample records by typology class, what the model sees at evaluation time | seconds | none beyond `requirements.txt` |
| [`02_baseline_reproduction.py`](02_baseline_reproduction.py) | Reproduce the paper's reported baseline F1 numbers end-to-end (random + zero-shot LLM + fine-tuned DeBERTa + paired bootstrap + IAA) | random: <5 min; zero-shot: 30 min – several hours; fine-tuned: ~75 min on Apple Silicon | llama.cpp servers for zero-shot; GPU/MPS for fine-tuned |
| [`03_error_analysis.py`](03_error_analysis.py) | Per-baseline confusion matrices, universally-hard cases, side-by-side baseline predictions on a specific pair, per-class recall breakdowns | <1 min | output of `02_baseline_reproduction.py` |

## Converting to Jupyter notebooks

```bash
pip install jupytext
jupytext --to ipynb notebooks/*.py
```

Then open the resulting `.ipynb` files in Jupyter / VS Code / etc. as normal.

## Running as scripts

The notebooks are self-contained Python — you can run them directly without converting:

```bash
python notebooks/01_dataset_exploration.py
```

This is the recommended path for CI and headless reproduction; convert to `.ipynb` only if you want interactive cell-by-cell exploration.

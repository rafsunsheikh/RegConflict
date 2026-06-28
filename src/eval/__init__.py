"""Workstream 6: evaluation pipeline.

Tasks (skill triag-ext-evaluation):
  * Task 1 — Retrieval (deferred; needs labelled retrieval questions)
  * Task 2 — Jurisdictional scoping (this slice)
  * Task 3 — Conflict detection (this slice; wraps WS3 outputs)
  * Task 4 — End-to-end briefing quality (deferred; needs gold briefings)

Each task module exposes:
  * a metrics function on per-example predictions + ground truth
  * (where needed) one or more system-config runners

The CLI ``scripts/run_eval.py`` wires this together with reproducibility
metadata (git commit, seed, timestamp, config snapshot).
"""

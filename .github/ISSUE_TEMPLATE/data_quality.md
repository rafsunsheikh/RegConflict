---
name: Data quality concern
about: Flag a labelling error, schema inconsistency, or source-document issue
title: "[data] "
labels: data-quality
---

## What records are affected

- `pair_id` (or `regime_id` / source document filename):
- Split (`train` / `val` / `test` / `gold_iaa`):

## Type of concern

- [ ] Suspected label error (binary or typology)
- [ ] Evidence chunk doesn't support the rationale
- [ ] Regime metadata incorrect
- [ ] Source document missing / unfetchable / SHA-256 mismatch
- [ ] Annotation guidelines need clarification on this case type
- [ ] Other (describe)

## Detail

(Be specific. Cite the relevant passage of the regulation, the rubric class that you believe should apply, and why.)

## Suggested resolution

(Optional. Re-label / clarify rubric / leave-as-is with explanatory note / remove from release / etc.)

## Sensitivity

- [ ] This concern reveals annotator-identifying information that should be redacted before public discussion
- [ ] This concern can be discussed publicly

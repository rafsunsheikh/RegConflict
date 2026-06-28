# Typology decision rubric

This rubric defines, in operational terms, when each of the four conflict-type labels applies. It is the authoritative source for how the labels in `data/conflicts/*.jsonl` were assigned. New annotators contributing to expansion releases should work from this rubric.

## Step 1 — Is this a conflict at all?

Before reaching for the typology, decide whether the pair is in conflict. Walk through this checklist:

1. **Can a single party simultaneously satisfy both regimes under reasonable assumptions?** If yes (and the satisfaction is straightforward), this is a non-conflict. Pick the appropriate `non_conflict_subtype` (see [`guidelines.md`](guidelines.md)).
2. **Is there a specific obligation in A that pushes against a specific obligation in B?** If no — i.e., the regimes operate in unrelated domains or impose layered but non-competing obligations — this is a non-conflict.
3. **Is the tension structural (in the rules as written), operational (in implementation), interpretive (depending on facts not in the text), or recurring (creating ongoing friction)?** If any of these four characterise the tension, this is a conflict; proceed to Step 2 to classify it.
4. **If you're not sure**, lean towards labelling as a conflict only when at least one specific obligation pair is identifiable. Topical proximity alone is not sufficient.

## Step 2 — Which conflict type applies?

The four classes are distinguished by their **resolution mechanism** — the route, or absence of one, by which an entity facing both regimes can comply.

```
                  ┌─────────────────────────────────────────┐
                  │ Is there a compliant intersection at    │
                  │ all? (Can both be satisfied in some     │
                  │ admissible state, even with effort?)    │
                  └────────────┬────────────────────────────┘
                               │ no              │ yes
                               ▼                 ▼
                  ┌────────────────────┐    ┌─────────────────────────┐
                  │ structural_        │    │ Can compliance with     │
                  │ unresolved         │    │ both be achieved via    │
                  │                    │    │ specific operational    │
                  │ Compliance with    │    │ mechanisms (legal       │
                  │ one prevents       │    │ instruments, contractual│
                  │ compliance with    │    │ provisions, technical   │
                  │ the other.         │    │ controls)?              │
                  └────────────────────┘    └────────────┬────────────┘
                                                         │ yes  │ no
                                                         ▼      ▼
                                          ┌─────────────────────────────┐
                                          │ operationally_resolvable    │
                                          │                             │
                                          │ Both can be satisfied via a │
                                          │ specific mechanism. Pick    │
                                          │ this class.                 │
                                          └─────────────────────────────┘
                                                                │
                                                                ▼
                                                  ┌─────────────────────────────┐
                                                  │ Does the conflict's outcome │
                                                  │ depend on specific facts    │
                                                  │ not determinable from       │
                                                  │ either regime's text alone? │
                                                  └─────────┬────────┬──────────┘
                                                            │ yes    │ no
                                                            ▼        ▼
                                          ┌───────────────────────────┐  ┌─────────────────────────┐
                                          │ interpretive_             │  │ recurring_friction      │
                                          │ fact_sensitive            │  │                         │
                                          │                           │  │ Both are operationally  │
                                          │ The conflict's status     │  │ satisfiable; no fact-   │
                                          │ turns on facts not        │  │ dependence; but the     │
                                          │ present in the passages   │  │ combination creates     │
                                          │ (e.g., marketing reach,   │  │ ongoing operational     │
                                          │ technical configuration). │  │ costs that recur.       │
                                          └───────────────────────────┘  └─────────────────────────┘
```

A version of this decision tree appears as Figure 3.3 in the paper. It is reproduced here verbatim so that annotators have the operational version of the schema, not just the paper's prose summary.

## Per-class definitions, with worked criteria

### `structural_unresolved`

> A conflict where current legal frameworks provide no clean path to satisfy both regulations simultaneously.

**Diagnostic test:**

- Is there a configuration in which an entity subject to both regimes can fully comply with each — even if effortful, expensive, or operationally awkward?
  - If **no** — the regimes have no compliant intersection — label `structural_unresolved`.
  - If **yes** — they're not structurally unresolved; check the next class.

**Examples (from the dataset):**

- **EU MiCA Title IV (EMT issuance) ↔ AU stablecoin authorisation.** A foreign-issued stablecoin cannot be issued as an EU EMT (Title IV requires EU establishment), nor can an entity authorised under one jurisdiction transit that authorisation to the other. There is no legal path for a single issuer to satisfy both regimes simultaneously without restructuring the entity.
- **EU MiCA Title V (CASP) ↔ Singapore PSA DPT licensing.** Parallel licensing requirements for crypto-asset service provision, with no recognition or passporting pathway between the two regimes.

**Common mistakes:**

- Labelling `structural_unresolved` when the conflict is actually `interpretive_fact_sensitive` (the structural impossibility only manifests *under certain factual conditions*). If facts can resolve the impossibility, this is interpretive.
- Labelling `structural_unresolved` when the conflict is `operationally_resolvable` via a known compliance instrument. The presence of any known resolution mechanism — SCCs, BCRs, exemptions, equivalence determinations — disqualifies this class.

### `operationally_resolvable`

> A conflict where both regulations can be satisfied via specific operational mechanisms (e.g., legal instruments, contractual provisions, technical controls).

**Diagnostic test:**

- Is there a specific, named operational mechanism — a legal instrument, contract type, technical control, or established compliance pathway — by which both regimes can be satisfied?
  - If **yes** — name the mechanism in the rationale — label `operationally_resolvable`.
  - If **no** — check `structural_unresolved` (no path at all) or `interpretive_fact_sensitive` (path depends on facts).

**Examples:**

- **EU AMLR/TFR (travel rule) ↔ EU GDPR.** Travel rule mandates data sharing with non-EU counterparties; GDPR restricts transfer of personal data outside the EEA. Resolution mechanism: Standard Contractual Clauses (SCCs), Chapter-V transfer instruments, recipient-jurisdiction adequacy assessments.
- **AU AML/CTF travel rule ↔ AU Privacy APP8.** Same shape as the EU TFR–GDPR conflict, in the AU context — resolved via lawful-basis chain and APP8-compliant transfer documentation.

**Common mistakes:**

- Naming a resolution mechanism that does not actually exist or does not actually cover the conflict (e.g., "SCCs cover this" when the data flow is into a jurisdiction not covered by valid SCCs). If the mechanism is hypothetical or contested, prefer `interpretive_fact_sensitive`.
- Labelling `operationally_resolvable` when the resolution is more accurately characterised as ongoing operational cost (a known friction that never fully resolves). That's `recurring_friction`.

### `interpretive_fact_sensitive`

> A conflict whose resolution depends on specific facts not yet determined, often involving definitions or scope boundaries.

**Diagnostic test:**

- Does the outcome of the conflict — whether it exists, how it resolves — turn on facts that the regulatory passages themselves do not determine? Examples of such facts: marketing reach ("offering"), technical configuration ("custody"), entity character ("professional client"), volume or frequency thresholds.
  - If **yes** — label `interpretive_fact_sensitive`.
  - If **no** — the conflict is either structurally unresolved or operationally resolvable; pick accordingly.

**Examples:**

- **EU MiCA Title V (CASP) ↔ reverse-solicitation doctrine.** Whether a service is being "offered" by a non-EU CASP or being "received at the client's own initiative" turns on facts about marketing reach, prior solicitation history, and platform availability — none of which are determinable from the regulatory text alone.
- **FATF VASP guidance ↔ ASIC crypto-asset regulation.** The boundary between regulated "exchange or transfer of virtual assets" and unregulated activities depends on the technical character of the platform — custody model, control over private keys, settlement architecture.

**Common mistakes:**

- Labelling `interpretive_fact_sensitive` when the fact-dependence is generic (every conflict depends on the facts of the entity). The fact-dependence has to be load-bearing: changing the facts changes the conflict.
- Confusing interpretive conflicts with disagreements between annotators about the law. If two annotators read the regulatory text differently, that's interpretive uncertainty in *labelling*, not interpretive fact-dependence in the *conflict itself*.

### `recurring_friction`

> Ongoing operational costs from competing requirements that don't fully resolve.

**Diagnostic test:**

- Can both regimes be satisfied operationally, but only by incurring costs (recordkeeping, transaction-level computation, dual reporting) that recur on every relevant event?
  - If **yes** — and there is no one-off legal instrument that closes the friction — label `recurring_friction`.
  - If **no** — if the friction is one-off (e.g., obtaining a single legal instrument once), it's `operationally_resolvable`.

**Examples:**

- **German tax treatment of crypto-asset acquisitions ↔ economic pass-through intent.** Each acquisition/disposal is a tax event in Germany even when the transaction is economically a pass-through. The friction recurs every time.
- **AUSTRAC threshold transaction reporting ↔ structuring prohibition.** The threshold creates an attractor for structuring behaviour; the prohibition exists because of the threshold. The friction recurs on every transaction near the threshold.

**Common mistakes:**

- Labelling `recurring_friction` when the situation is actually `operationally_resolvable` once and then settled. If the cost is paid once (one-off SCC drafting, one-off licensing), it's not recurring.
- Labelling every "ongoing compliance cost" as `recurring_friction`. The cost has to arise specifically from the *combination* of regimes, not from either regime alone.

## What if more than one class seems to apply?

A small set of conflicts genuinely span more than one class — e.g., a conflict that is operationally resolvable for most entities but fact-sensitive for a subset. The annotator's job is to pick the **dominant** class for the conflict as labelled.

The tiebreaker rule, applied in this order:

1. If there's no compliant intersection at all → `structural_unresolved`.
2. Else, if the outcome turns on specific facts not in the passages → `interpretive_fact_sensitive`.
3. Else, if both regimes can be satisfied via a named operational mechanism → `operationally_resolvable`.
4. Else, if the combination creates ongoing recurring cost → `recurring_friction`.

The rationale should note the second-best class when one is plausible; this aids future re-annotation and downstream multi-label experiments.

## Severity

Each conflict additionally carries a `severity` label ∈ {`low`, `medium`, `high`}. Severity is *not* a function of the typology — a `recurring_friction` conflict can be high-severity if the recurring cost is large, and a `structural_unresolved` conflict can be low-severity if the affected entity class is narrow.

The severity rubric:

- **`high`**: structurally significant; affects core business viability for any entity subject to both regimes (e.g., the entity cannot operate in one of the two jurisdictions without restructuring).
- **`medium`**: meaningful operational impact requiring active compliance effort but not blocking business viability (most travel-rule × privacy conflicts fall here).
- **`low`**: marginal operational impact; the conflict exists but is resolvable with routine compliance posture.

## Inter-annotator agreement guarantees and limits

The 30-pair gold IAA subset of conflict cases was independently labelled by two expert annotators after a rubric-calibration round on a disjoint training set. Cohen's κ on the four-class typology task was κ = 0.6 — substantial agreement on the Landis–Koch (1977) scale, indicating that the four-class schema is recoverable by a second expert annotator to a substantial-agreement level after rubric calibration. Annotators should mark borderline cases (those near class boundaries) with the second-best class noted in the rationale. The full IAA records and the κ-computation script (`scripts/compute_iaa.py`) are released for reproducibility.

## Where this rubric is implemented

- The typology enum is defined in `src/eval/benchmark/schema.py`.
- A class-distribution audit script lives at `src/tools/verify_corpus.py` and reports per-class counts on every split.
- Worked examples (3 per class) drawn from the test data appear in `docs/annotation_protocol.md` and in the paper appendix.

# TrajRel Experimental Plan

## Status

This document defines the experimental structure before the main comparative
evaluation.

Development evidence may be used to design and calibrate the method.
Held-out evaluation data must not be used to modify the method after freeze.

---

# 1. Core research hypothesis

Let:

- B_t = structured identifiers supported by prior grounded trajectory evidence
- Q_t = structured identifiers explicitly reused in the agent's current action
- A_t = B_t ∩ Q_t

Primary hypothesis:

Explicit re-adoption of previously grounded trajectory-local state is a
high-precision signal for selective preservation under lossy context
compression.

TrajRel is primarily evaluated as a preservation layer over an existing
compressor, not as a claim to be a universally superior standalone compressor.

---

# 2. Main research questions

## RQ1 — Historical bridge scoring

Which trajectory-evidence signals are actually useful for identifying
historically supported bridge identifiers?

Reference scorer signals:

- causal evidence
- cross-observation corroboration
- occurrence
- recency
- speculation penalty

The frozen KTH implementation is a reference baseline, not assumed optimal.

## RQ2 — Preservation rule

Given historical support B and current reuse Q, how do the following policies
compare?

- Base
- History-only: B
- Lexical query-only: Q_lexical
- Semantic query relevance: Q_semantic
- Adopted bridge: B ∩ Q
- Random budget-matched preservation
- Recency budget-matched preservation

Primary trade-off:

critical evidence retained versus non-critical context added.

## RQ3 — Complementarity with existing compressors

Does TrajRel improve existing context-management systems when applied as an
overlay?

Compare, where technically compatible:

- Base
- Base + TrajRel
- Squeez
- Squeez + TrajRel
- AttnCompress
- AttnCompress + TrajRel

Other systems such as SWE-Pruner, LaMR, ACON, CoACT, OSU-Mem-style overlap
methods, or related approaches should be included where the task/input
representation permits an apples-to-apples comparison.

Do not force incompatible methods into the same benchmark merely to increase
the number of baselines.

## RQ4 — Agent recovery cost

If critical evidence is discarded, can a strong coding agent reconstruct it?

If yes, what is the cost?

Measure:

- repeated searches
- additional tool calls
- additional input tokens
- action/reasoning steps
- wall-clock time
- monetary cost
- final task success
- patch correctness

A zero task-success difference does not imply zero effect if recovery cost
changes materially.

---

# 3. Evaluation levels

## Level A — Mechanism ablation

Use development data only.

Evaluate scorer components independently.

### Score-term ablations

- Full reference scorer
- no causal score
- no cross-observation score
- no occurrence score
- no recency score
- no speculation penalty

Important:

Score-term ablations must be distinguished from eligibility/gating ablations.
For example, causal evidence is used both in scoring and in candidate
eligibility. Setting causal_weight=0 does not fully remove causal evidence.

Therefore later experiments must separately evaluate:

### Eligibility/gate ablations

- no positive-evidence eligibility
- no cross-observation eligibility
- no weak-identifier filter
- no generic-identifier filter
- no singleton corroboration
- no target-specificity adjustment
- no user-context duplicate exclusion

Leakage-prevention constraints such as exclusion of target self-evidence are
not treated as optional performance tricks.

---

# 4. Signal-level policy evaluation

All policies must operate on exactly the same compression opportunities.

Compare:

1. Base
2. Random preservation, token-budget matched
3. Recent-context preservation, token-budget matched
4. History-only B
5. Lexical Q
6. Semantic Q
7. B ∩ Q
8. Semantic Q + B ∩ Q

The final hybrid is important because TrajRel may be complementary to semantic
relevance rather than strictly superior to it.

---

# 5. Core mechanism metrics

For every policy report:

- total critical records
- critical records retained
- critical recall
- baseline critical drops
- critical DROP -> KEEP rescues
- KEEP -> DROP regressions
- newly forced keeps
- critical newly forced keeps
- non-critical newly forced keeps
- incremental preservation precision
- added critical tokens
- added non-critical tokens
- total added tokens
- compression ratio / compression overhead

Incremental precision is defined over newly preserved records:

    forced_critical / forced_total

rather than over the base compressor's inherited KEEP set.

---

# 6. Fairness regimes

Do not compare methods only at their default thresholds.

## Fixed-token-budget comparison

Give competing preservation methods the same additional context budget.

Example:

- random: +1000 tokens
- semantic: +1000 tokens
- B-only: +1000 tokens
- B ∩ Q: +1000 tokens

Then compare critical evidence retained.

## Fixed-recall comparison

For a target critical recall, measure how many additional tokens each method
requires.

Example:

    tokens required to reach 90% critical recall

## Pareto analysis

Plot:

    x = additional non-critical / retained tokens
    y = critical evidence recall

Compare complete threshold/budget curves rather than one cherry-picked
operating point.

---

# 7. Development and held-out separation

Existing Headroom artifacts are development/exploratory evidence.

They include:

- controlled mechanism tests
- matched replay experiments
- historical reverse-patch analysis
- prior KTH experiments

They must not be presented as final generalization evidence.

After development:

1. freeze extraction logic
2. freeze scorer
3. freeze selector
4. freeze preservation policy
5. freeze eligibility definition
6. freeze evaluation code
7. tag the repository version

Only after this freeze may final held-out trajectories be evaluated.

No tuning is allowed on held-out outcomes.

---

# 8. Held-out trajectory collection

Target:

- multiple repositories
- multiple task types
- multiple tool-output formats
- approximately 50–100+ real tasks if feasible
- several hundred eligible compression opportunities

Eligibility must be defined without using the outcome of TrajRel.

A checkpoint must not be selected merely because B ∩ Q is known to activate.

Labels for held-out critical evidence should be produced independently from
the scorer-development process where feasible.

---

# 9. Forked-trajectory counterfactual evaluation

For an eligible real compression point:

1. checkpoint the same agent state before the compression decision
2. fork execution
3. keep model, task, repository, tools, temperature/settings, context history,
   and budgets identical
4. change only the context-management treatment

Primary pair:

    base compressor
    base compressor + TrajRel

Additional paired conditions when feasible:

    base + random same-budget preservation
    base + semantic preservation
    semantic compressor
    semantic compressor + TrajRel

For external compressors:

    Squeez
    Squeez + TrajRel

    AttnCompress
    AttnCompress + TrajRel

Use paired statistical analysis.

---

# 10. End-to-end metrics

Report:

- task success
- patch/test correctness
- total input tokens
- total output tokens
- total tool calls
- repeated/redundant searches
- recovery searches
- number of agent steps
- wall-clock time
- monetary cost

Task success is not the only primary outcome.

Recovery cost is a core research outcome.

---

# 11. Sensitivity analysis

Evaluate robustness to:

- bridge top-k
- minimum scorer threshold
- token budget
- identifier filtering
- target-specificity strength
- semantic-relevance threshold

Do not report only the single setting that maximizes the desired result.

---

# 12. Interpretation discipline

Do not claim:

- that trajectory-aware compression itself is novel
- that monotonicity is an empirical contribution
- that parity with the KTH implementation proves algorithmic quality
- that development/post-hoc evidence proves generalization
- that one critical rescue establishes effectiveness

Potential contribution:

Characterizing whether the intersection of prior grounded support and later
explicit agent reuse provides a cheap, interpretable, high-precision
preservation signal, and whether that signal complements stronger semantic or
learned context-management systems.

A negative result is valid.

Possible negative findings include:

- explicit reuse activates too rarely
- semantic relevance dominates B ∩ Q under equal budgets
- B ∩ Q loses too much recall
- one scorer component contributes nothing
- strong agents reconstruct removed evidence at negligible cost

These outcomes must be reported rather than tuned away.

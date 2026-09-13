# TrajRel Held-Out Evaluation Protocol

## Purpose

The development corpus showed that the exact B∩Q preservation rule can
rescue critical evidence with low incremental context cost, but all observed
base-dropped critical opportunities were concentrated in one task cluster.

The held-out evaluation therefore tests task-level generalization rather than
further tuning the development result.

## Frozen primary method

For each compression opportunity:

1. Use only tool outputs strictly before the current agent action.
2. Rank historical structured identifiers with the frozen Headroom reference
   scorer.
3. Apply the frozen target-conditioned selector with top-k 6.
4. Let B be the selected historical bridge identifiers.
5. Let Q be conservative structured identifiers explicitly reused in the
   current rg/grep search pattern.
6. Let A = B ∩ Q.
7. Preserve the base compressor decision monotonically:
   - base KEEP -> KEEP
   - base DROP + exact-boundary match with any identifier in A -> KEEP
   - otherwise DROP

No held-out task result may be used to change these rules before the primary
evaluation is complete.

## Primary unit of evidence

A useful opportunity requires:

- critical = true
- base compressor = DROP
- record is scorable

Critical records already kept by the base compressor do not constitute rescue
opportunities.

## Primary claims

The evaluation may support claims about:

- critical D->K rescue,
- incremental precision,
- token overhead,
- task-level breadth of rescue evidence.

It must not claim held-out generalization if positive rescue evidence is
concentrated in a single task.

## Development-only variants

The explicit action-context envelope and semantic/tool-aware Q are exploratory.
They are not part of the preregistered held-out primary analysis.

## Pilot criterion

Collect 10-20 unseen tasks first.

Do not scale to the full evaluation until the pilot establishes that multiple
tasks contain base-dropped critical opportunities.

The pilot is informative even if TrajRel produces zero rescues: the key first
question is whether the evaluation set contains real opportunities.

## Full evaluation target

If the pilot is viable:

- 50-100 held-out tasks,
- multiple task families,
- multiple opportunity-bearing tasks,
- preferably more than one compressor,
- task-level and record-level reporting.

# TrajRel

### Trajectory-conditioned relevance for selective context preservation in coding agents

**TrajRel** is an experimental framework for studying whether information that becomes important later in an agent trajectory should receive additional protection during context compression.

The project investigates a simple but consequential failure mode of long-horizon coding agents:

> A piece of information may look irrelevant when it is first produced, but become important later because the agent explicitly reuses it.

Standard relevance-based compression evaluates context largely from the current request. TrajRel adds a lightweight trajectory signal: **what the agent has previously learned and what it is explicitly reusing now**.

The project was developed and evaluated as an extension/integration study around [Headroom](integrations/headroom/), with the goal of testing whether trajectory-aware preservation can improve information fidelity while retaining meaningful compression.

> **Research status:** early-stage research / proof of concept.
> The experiments demonstrate the proposed mechanism and provide replay-based evidence, but do **not** establish a general end-to-end task-success improvement.

---

## Research question

Context compression is necessary for long-running LLM agents, but aggressive compression can remove information that later becomes useful.

Consider an agent that:

1. searches a repository,
2. discovers a function or configuration identifier,
3. continues working for several turns,
4. later searches for that identifier explicitly.

At the moment the original tool output is compressed, the identifier may not appear highly relevant to the current request. A purely local relevance signal can therefore discard it.

TrajRel asks:

**Can trajectory information be used as a conservative preservation signal, so that information previously established by the agent and explicitly reused later is less likely to be dropped?**

The objective is not to disable compression. Instead, the design attempts to preserve only a small, evidence-backed subset of potentially important historical information.

---

# Core idea

TrajRel introduces a trajectory-conditioned preservation layer on top of a base compressor.

Let:

* **Bₜ** = historically supported *bridge identifiers* available from the previous trajectory
* **Qₜ** = identifiers explicitly reused in the current action/query
* **Aₜ = Bₜ ∩ Qₜ** = adopted bridge identifiers

The key policy is therefore:

```text
historical evidence ∩ current explicit reuse
                    ↓
             adopted bridges
                    ↓
          conservative preservation
```

An identifier is not protected merely because it appeared somewhere in history.

It becomes a preservation candidate when:

1. it was previously observed and supported by trajectory evidence,
2. its provenance can be traced,
3. it is explicitly reused by the current action,
4. the resulting relevance signal survives the adoption rules.

This is intentionally more conservative than simply keeping everything mentioned earlier.

---

# Preservation policy

TrajRel is designed to be **monotonic with respect to the underlying compressor**.

The trajectory layer can:

```text
DROP → KEEP
```

but does not turn an existing:

```text
KEEP → DROP
```

This means TrajRel acts as a conservative preservation floor rather than replacing the base compression decision.

Conceptually:

```text
                 Base Compressor
                       │
                       ▼
              initial relevance
                       │
                       ▼
             trajectory evidence
                       │
                       ▼
              bridge extraction
                       │
                       ▼
            current-query adoption
                       │
                       ▼
             Bₜ ∩ Qₜ preservation
                       │
                       ▼
             final KEEP / DROP
```

The implementation also uses **exact-boundary preservation**, so the trajectory policy can rescue information at the compression boundary without globally relaxing the compressor.

---

# What we implemented

The repository contains the research implementation and the experiments used to investigate the hypothesis.

## 1. Bounded trajectory state

Rather than retaining the entire previous trajectory, TrajRel maintains a bounded representation of historically useful identifiers.

This keeps the trajectory signal compatible with context compression instead of creating another unbounded memory mechanism.

The state is designed around:

* previously observed identifiers,
* their provenance,
* supporting tool outputs,
* relevance/corroboration information,
* current reuse signals.

---

## 2. Structured bridge extraction

We introduced structured extraction of **trajectory bridges** from previous tool outputs.

A bridge is an identifier that can connect an earlier observation to a later action, such as:

* function names,
* class names,
* configuration keys,
* file paths,
* commands,
* symbols,
* search terms,
* other explicitly reusable technical identifiers.

Bridge candidates are tracked with provenance rather than being treated as anonymous strings.

This makes the mechanism inspectable and allows the experiments to distinguish:

```text
observed historically
        ≠
supported historically
        ≠
explicitly reused now
        ≠
actually adopted
```

---

## 3. Bridge ranking and corroboration

Candidate bridges are not adopted indiscriminately.

The implementation uses trajectory evidence and corroboration to determine which identifiers are sufficiently supported to participate in the preservation policy.

This was important during development because simply relaxing eligibility thresholds caused too many weak candidates to be retained.

The resulting design therefore favors:

**specific trajectory evidence over global threshold relaxation.**

---

## 4. Query-conditioned adoption

The central adoption rule is:

```text
Aₜ = Bₜ ∩ Qₜ
```

An identifier that was historically supported becomes a stronger preservation candidate when the agent explicitly reuses that identifier in the current action.

For example:

```text
Earlier tool output:
    class SearchIndexer

Later agent query:
    "find where SearchIndexer is initialized"

                     ↓

              explicit reuse
                     ↓

          adopted bridge candidate
                     ↓

             KEEP floor applied
```

This makes trajectory relevance conditional on the agent's actual behavior rather than on hindsight alone.

---

## 5. Relevance split and KEEP floor

TrajRel separates the ordinary compression decision from the trajectory-derived preservation signal.

The resulting **relevance split** allows us to measure when trajectory-aware relevance actually changes a compression decision.

The **KEEP floor** then protects adopted trajectory bridges at the compression boundary.

This gives us an observable distinction between:

* information that the base compressor already keeps,
* information rescued by TrajRel,
* information that remains dropped,
* and cases where the trajectory mechanism abstains.

That distinction is important for evaluating whether the mechanism actually contributes anything beyond the base compressor.

---

# Experimental program

We deliberately evaluated TrajRel using multiple forms of evidence rather than relying on a single benchmark.

The experiments are separated into:

1. **controlled mechanism validation**
2. **natural agent runs**
3. **hard-matched replay**
4. **historical replay**
5. **prospective / held-out evaluation**
6. **negative and reachability analysis**

This separation is important because historical replay can provide useful evidence about preservation while still suffering from hindsight/oracle bias.

---

# Main experimental results

## Controlled mechanism validation

In the controlled evaluation, the trajectory-aware policy improved the measured fidelity criterion from:

| Metric              |       Base |     TrajRel |
| ------------------- | ---------: | ----------: |
| Controlled fidelity | **6 / 20** | **20 / 20** |
| Compression         | **25.46%** |  **19.13%** |

The important observation is that the mechanism did not achieve the fidelity improvement by simply disabling compression.

Compression remained active, but the trajectory-aware policy preserved information at selected boundaries.

In other words:

> The controlled experiment validates that the proposed preservation mechanism can change compression decisions in the intended direction while retaining substantial compression.

---

## Strict adopted-bridge preservation

Under the stricter adopted-bridge criterion:

| Metric                         |      Base |   TrajRel |
| ------------------------------ | --------: | --------: |
| Critical retention             | **5 / 9** | **6 / 9** |
| Additional non-critical tokens |         — |  **+504** |

For comparison, a naive broader preservation strategy would have introduced approximately:

**+3,249 non-critical tokens**

The result supports the intended design principle: use **specific adopted trajectory evidence** instead of globally relaxing relevance thresholds.

---

# Historical replay

We performed an exhaustive historical replay over the available trajectory corpus.

The scan produced:

* **2,398** non-merge commits examined
* **570** validated candidates
* **105** evaluable candidates
* **92** baseline-opportunity tasks
* **10** rescue tasks
* **8** fresh cases
* **2,342** critical records
* **1,326** baseline DROP decisions
* **60** DROP → KEEP changes
* **3** KEEP → DROP cases investigated

The main preservation signal was:

**60 / 1,326 baseline drops were rescued.**

There were also **3 regression cases**, which were retained in the analysis rather than excluded.

This is important: the replay result should be interpreted as **conditional preservation evidence**, not as proof that agents solve more tasks.

Historical replay can identify cases where the trajectory mechanism would have preserved information that the baseline discarded, but the replay itself cannot establish that the rescued information would necessarily have changed the agent's final outcome.

---

# Hard-matched replay

We also constructed a harder matched replay designed to reduce ambiguity around whether the relevant trajectory context actually existed.

The replay contained:

* **13 valid units**
* **18 critical records**

The baseline exhibited:

* **0 / 18** critical recall under the relevant baseline condition
* trajectory context present in **12 / 14** analyzed units in the corresponding earlier analysis
* meaningful baseline/trajectory differences only in a subset of the units

The broader replay analysis produced:

* **60** DROP → KEEP rescues
* **3** regression cases

These experiments were useful for testing the mechanism under controlled replay conditions, but they are not presented as end-to-end agent-success results.

---

# Natural agent runs

Natural agent execution was used to test whether the trajectory mechanism could actually become reachable through realistic agent behavior.

This was one of the most important findings of the project:

**having a theoretically correct trajectory mechanism is not sufficient if the live agent/tool trajectory never produces an eligible adopted bridge.**

Several natural runs passed their underlying tasks while producing:

```text
relevance_split_units = 0
```

For example, early natural ON runs included successful task completion without trajectory relevance activation.

We therefore treat:

```text
task PASS
```

and:

```text
trajectory mechanism activated
```

as separate measurements.

This prevents ordinary agent success from being incorrectly attributed to TrajRel.

---

# Activation and negative results

Activation analysis exposed several practical limitations in the initial integration.

Some trajectories contained relevant-looking Bash/`rg` outputs but did not produce an adopted bridge.

Other trajectories reached the SEARCH route without generating a usable candidate.

This led to an important engineering finding:

> The main bottleneck was not necessarily the preservation policy itself; it could occur earlier in the trajectory-to-bridge pipeline.

In particular, we identified cases involving:

* incomplete Bash/`rg` output routing,
* search results that were routed correctly but produced no eligible candidate,
* newly discovered identifiers without sufficient corroboration,
* activation that could not be inferred from a single `act=False` field,
* trajectory context existing without being adopted.

These negative results are part of the research record and are preserved in the repository.

---

# Prospective evaluation

A prospective OFF/ON evaluation was also constructed rather than relying only on historical replay.

The early prospective set included the N5–N8 conditions:

* **4 / 4** task completion in the corresponding OFF/ON evaluation,
* **3 / 4** cases with observable trajectory activation.

A harder replay subsequently achieved:

* **6 / 6** valid cases.

These results demonstrate that the mechanism can be exercised prospectively under controlled conditions, but the sample is too small to support a general claim about end-to-end agent performance.

---

# Held-out pilot

To avoid tuning directly against the final evaluation set, a held-out pilot protocol was frozen.

The frozen pilot contains **10 OFF conditions**:

```text
U04
U05
U06
U07
U08
U09
U10
U11
U12
P10
```

The execution protocol, method locks, and snapshots were frozen before the complete OFF pass.

The OFF trajectory collection was completed:

```text
10 / 10 conditions
```

with the raw artifact lock containing **12,558 files** and checksum:

```text
0408d440…bf50a
```

The held-out evaluation is intentionally kept separate from the earlier exploratory experiments.

This distinction prevents exploratory tuning results from being presented as independent held-out evidence.

---

# What the experiments show

The evidence currently supports several narrower conclusions.

### Supported

* Trajectory information can be represented as bounded structured state.
* Previously observed identifiers can be tracked with provenance.
* Explicit current reuse can be used to adopt a subset of historical bridges.
* The adopted-bridge policy can conservatively modify a base compressor.
* The policy can rescue selected baseline DROP decisions.
* The controlled experiment demonstrates the intended DROP → KEEP mechanism.
* Historical replay provides measurable conditional preservation evidence.
* The mechanism can be integrated into realistic Headroom compression flows.
* Activation/reachability can be instrumented independently from task success.

### Not established

We do **not** claim that TrajRel:

* universally improves coding-agent task success,
* improves benchmark performance across arbitrary agents,
* solves long-context compression,
* eliminates information loss,
* is optimal compared with all alternative relevance mechanisms,
* provides a statistically established end-to-end performance improvement.

The current evidence is best described as:

> **mechanism validation + replay evidence + systems integration, with early prospective evaluation.**

---

# Engineering architecture

The repository separates the generic trajectory-relevance mechanism from the Headroom integration.

```text
trajrel/
│
├── benchmarks/
│   └── Evaluation and benchmark artifacts
│
├── experiments/
│   └── Controlled, replay, and analysis experiments
│
├── integrations/
│   └── headroom/
│       └── Headroom integration layer
│
├── src/
│   └── trajrel/
│       └── Generic trajectory-relevance policy
│
├── tests/
│   └── Unit and behavioral tests
│
└── docs/
    └── Research notes, protocols, and experiment documentation
```

The core implementation is intentionally separated from the Headroom-specific integration so that the trajectory relevance policy can be studied independently from a particular compression backend.

---

# Headroom integration

TrajRel was designed around the context-compression setting exposed by Headroom.

Headroom provides compression of agent context/tool outputs before they reach the model; TrajRel adds a trajectory-conditioned preservation signal on top of that compression pipeline. Headroom's current architecture similarly treats compression as a pipeline with different content-aware transforms and integration points.

The integration therefore focuses on:

```text
Agent trajectory
      │
      ├── previous tool outputs
      │
      ▼
Trajectory state
      │
      ├── bridge extraction
      ├── provenance
      ├── corroboration
      └── bounded history
      │
      ▼
Current action/query
      │
      ▼
Bₜ ∩ Qₜ
      │
      ▼
Trajectory relevance
      │
      ▼
Headroom compression decision
      │
      ▼
Preserved context
```

The implementation does not require replacing the underlying compressor with a completely different compression algorithm.

Instead, it introduces an additional conservative signal.

---

# Why the B ∩ Q design matters

A naive trajectory-aware compressor could simply say:

> "This identifier appeared earlier, therefore keep it."

That approach quickly becomes too permissive.

TrajRel instead asks two questions:

### 1. Did the trajectory establish this information?

```text
Bₜ
```

### 2. Is the agent explicitly using it now?

```text
Qₜ
```

Only their intersection becomes:

```text
Aₜ = Bₜ ∩ Qₜ
```

This creates a much narrower preservation policy.

The design therefore tries to capture **trajectory relevance**, rather than simply adding another long-term memory layer.

---

# Reproducibility

A major goal of the repository is to make the experimental process inspectable.

The repository contains:

* implementation code,
* benchmark definitions,
* integration code,
* tests,
* experiment scripts,
* frozen evaluation artifacts,
* trajectory/replay analysis,
* protocol documentation.

Where possible, experiments distinguish:

```text
exploratory
    ↓
controlled
    ↓
replay
    ↓
prospective
    ↓
held-out
```

rather than mixing all evidence into a single headline number.

This is particularly important for trajectory-aware methods because retrospective knowledge can easily leak into an apparently "relevance" based decision.

---

# Limitations

## 1. Historical replay has hindsight bias

Historical replay knows what happened later.

Therefore, it can answer:

> "Would this information have been preserved under the proposed policy?"

but not necessarily:

> "Would preserving it have caused the agent to succeed?"

For this reason, historical replay is treated as supporting evidence rather than end-to-end performance evaluation.

---

## 2. Activation is sparse

A trajectory-aware mechanism is useful only when the trajectory produces an eligible signal.

Several natural runs completed successfully without activating the trajectory relevance split.

This shows that **mechanism correctness and mechanism reachability are different problems**.

---

## 3. Bridge extraction is heuristic

The current implementation relies on structured identifiers and corroboration rather than a learned universal semantic relevance model.

This makes the system interpretable, but also limits generalization.

---

## 4. Small prospective evaluation

The prospective and held-out experiments are still small.

The current results should therefore be viewed as evidence that motivates larger evaluation, not as a final statistical benchmark.

---

## 5. No end-to-end uplift claim

Although the mechanism demonstrates controlled fidelity improvements and replay-based rescues, we deliberately do not claim an end-to-end task-success improvement from the current experiments.

That is a future evaluation target.

---

# Research contribution

The main contribution of this repository is not a claim that "trajectory-aware compression is solved."

Instead, the project contributes an experimentally testable formulation of trajectory relevance:

1. **Bounded trajectory state** for compression-compatible historical context.
2. **Structured bridge extraction** with provenance.
3. **Bridge corroboration/ranking** to avoid indiscriminate historical retention.
4. **Query-conditioned adoption** through `Bₜ ∩ Qₜ`.
5. **Monotonic DROP → KEEP preservation** over an existing compressor.
6. **A measurable relevance split** separating base compression from trajectory-induced preservation.
7. **Controlled, replay, prospective, and held-out evaluation protocols.**
8. **Negative/reachability analysis** showing where the mechanism fails to activate.
9. **A concrete Headroom integration** demonstrating how trajectory relevance can be incorporated into an existing context-compression system.

The repository therefore serves both as a research prototype and as an experimental record of what works, what does not, and where the remaining uncertainty lies.

---

# Repository structure

| Path                                             | Purpose                                         |
| ------------------------------------------------ | ----------------------------------------------- |
| [`src/trajrel`](src/trajrel)                     | Generic trajectory-relevance implementation     |
| [`integrations/headroom`](integrations/headroom) | Headroom integration                            |
| [`experiments`](experiments)                     | Controlled, replay, and prospective experiments |
| [`benchmarks`](benchmarks)                       | Benchmark/evaluation artifacts                  |
| [`tests`](tests)                                 | Unit and behavioral tests                       |
| [`docs`](docs)                                   | Research notes, protocols, and analysis         |

---

# Status

**Research prototype — October 2026**

Current work focuses on:

* improving trajectory-bridge reachability,
* reducing heuristic dependence,
* expanding prospective evaluation,
* testing broader agent/tool trajectories,
* quantifying the trade-off between preservation and compression,
* and determining whether trajectory-aware preservation produces measurable downstream task improvements.

---

# Acknowledgment

This work studies trajectory-conditioned relevance in the context of [Headroom](https://github.com/headroomlabs-ai/headroom), an open-source context-compression system for LLM applications and agents.

TrajRel is an independent research implementation and experimental repository focused specifically on the trajectory-relevance hypothesis and its evaluation.

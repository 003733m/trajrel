# TrajRel

Trajectory-conditioned relevance and selective context preservation for agents.

TrajRel studies whether information learned earlier in an agent trajectory,
and explicitly reused later by the agent, should receive a conservative
preservation floor during context compression.

Core notation:

- B_t: historically supported bridge identifiers
- Q_t: identifiers explicitly reused in the current action
- A_t = B_t ∩ Q_t: adopted bridge identifiers

The adopted-bridge policy is monotonic with respect to the base compressor:
it may convert DROP to KEEP, but never converts an existing KEEP to DROP.

Status: early research implementation.

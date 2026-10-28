---
name: decisive-experiments
description: "The user only accepts experiments with sharp, causal, binary outcomes; no eval-time interventions, no graded knobs"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ae142cb7-3355-47e0-b126-2adf442a38bb
  modified: 2026-10-27T07:59:21.639Z
---

Experiments must be interventions with a clear yes/no causal reading, trained from scratch (e.g. direct output path present vs absent; evaluation tasks' demos in training vs held out; fixed vs freshly generated pairs).

**Why:** the user (2026-10-27) rejected eval-time interventions on trained models as out-of-distribution (a lambda sweep at evaluation only showed collapse), and rejected graded knobs like dropout rates or width reductions because their boundaries are blurry and prove no causation.
**How to apply:** before proposing a run, state the single question it answers and how each outcome would be read; drop anything whose result would only be "a bit better/worse". Pair with [[no-multi-seed]].

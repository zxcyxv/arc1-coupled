---
name: early-generalization-gap
description: "User's view that recursive ARC models (TRM-like, coupled) show a large train-vs-eval gap early; don't alarm on early quick evals"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ae142cb7-3355-47e0-b126-2adf442a38bb
  modified: 2026-10-25T13:59:50.390Z
---

Early in training, a big gap between training cell/exact and evaluation exact is expected for this model family (TRM repro: eval cell 0.688 but exact 0.004 at 10k epochs; 5090 harness: eval cell 0.802, exact 1/419 at 6,224 steps).

**Why:** the user pointed this out on 2026-10-25 before the first quick eval of the 4xH100 ARC-1 run.
**How to apply:** don't treat a low early quick exact as a problem or propose changes from it; judge only by the agreed rule ([[arc1-switch-rule]]) at 10k+ epoch vote evals.

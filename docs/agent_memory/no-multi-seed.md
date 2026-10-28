---
name: no-multi-seed
description: The user forbids running multiple seeds of the same configuration; design experiments as single decisive runs
metadata:
  node_type: memory
  type: feedback
  originSessionId: ae142cb7-3355-47e0-b126-2adf442a38bb
  modified: 2026-10-27T07:59:18.154Z
---

Do not propose or run several seeds of the same configuration (the user said "seed 여러개는 금지야" on 2026-10-27).

**Why:** the user wants every GPU-minute spent on experiments that discriminate hypotheses; seed replicates only measure noise.
**How to apply:** design single runs whose outcomes are binary and causally interpretable (clear interventions on data or structure, trained from scratch, see [[decisive-experiments]]); judge differences against the known run-to-run noise from past records instead of new replicates.

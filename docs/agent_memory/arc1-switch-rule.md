---
name: arc1-switch-rule
description: "2026-10-25 ARC-1 main run plan on 4xH100: compare to TRM at 10k/15k/20k epochs, switch to 10 blocks all-backprop if behind; Claude monitors alone with standing push/NaN authority"
metadata:
  node_type: memory
  type: project
  originSessionId: ae142cb7-3355-47e0-b126-2adf442a38bb
  modified: 2026-10-25T13:56:55.010Z
---

On 2026-10-25 (~13:30 UTC) the user started being away for ~19 hours and made Claude responsible for monitoring the ARC-1 run (`runs/arc1`) directly.

Rule (also in docs/decisions.md, row "ARC-1 전환 규칙"): compare vote `view_exact` to TRM repro `all/exact_accuracy` (10k 0.004, 15k interp 0.029, 20k 0.054). At 15k, behind at both 10k and 15k → `run.sh stop`, skip the final eval, start `runs/arc1_b10` with `model.blocks=10 model.grad_blocks=10`, 50k epochs, GA from a fresh bench. At 20k, behind at 15k and 20k → same switch. Otherwise continue ARC-1 to 50k.

Standing authority given 2026-10-25: (1) at every vote eval, `run.sh archive` + commit + push (incl. latest full checkpoint); (2) on NaN/non-finite stop, Claude fixes it as it sees fit and resumes (record any recipe deviation in docs/decisions.md and tell the user).

No second instance is possible: the 4xH100 SXM at ~$7/h was a rare listing (usual ~$13/h), so ARC-2 must run serially on this instance.

**Why:** time is the binding constraint (paper due ~2026-11-01); the user wants an early decision instead of finishing a lagging run.
**How to apply:** keep a background watcher armed on runs/arc1 at all times; act on the rule without waiting for the user; report at vote evals and errors. 10 blocks all-backprop is expected to be fine: the 4-block all-backprop ablation lagged because 4 total iterations don't reach the fixed point, not because no-grad blocks were absent.

Update 2026-10-26 (KST evening): the switch rule above is superseded. ARC-1 now runs as `runs/arc1_b16g10` (16 blocks, last 10 backprop, 10k epochs, final eval 8 versions). A detached script (scratchpad chain_arc2.sh, log chain_arc2.log) starts `runs/arc2_b16g10` (ARC-2, 16/10, 10k epochs, 1001-version final eval) as soon as ARC-1 finishes normally. Credits (no top-up) end about 2026-10-27 21:30 KST; ARC-2 is tight, so watch its ETA and shrink the final eval if needed (ask the user).

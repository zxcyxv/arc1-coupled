---
name: paper-thesis
description: "ARC Prize 2026 paper thesis agreed 2026-10-28: STDP-over-recursion (main) + TEM factorization; what was dropped, evidence sources, gaps"
metadata:
  node_type: memory
  type: project
  originSessionId: ae142cb7-3355-47e0-b126-2adf442a38bb
  modified: 2026-10-28T07:15:08.171Z
---

Agreed with the user 2026-10-28 (paper track, ARC Prize 2026; paper due 11-08, needs a working ARC-AGI-2 Kaggle code submission by 11-02; rubric: Accuracy, Universality, Progress, Theory, Completeness, Novelty, equal weight, 0-5 each).

**Main thesis (Theory + Novelty):** Hebbian -> fast weight -> linear attention lineage inherits the timeless formal Hebbian rule (Hebb's postulate was causal; STDP restores the timing). Recurrent transformers (HRM/TRM/URM) create a time axis (recursion) but keep no memory across it; accumulating same-time Hebbian memory across recursion self-reinforces the current (possibly wrong) attractor. A temporally asymmetric learned kernel over recursive depth (STDP-like; its DC part is the Hebbian term, so it generalizes Hebb, not replaces it) integrates relations between different times; its odd part vanishes at fixed points and encodes the trajectory.
User corrections 10-28: Hebbian does not accumulate over time by definition, so NO "Hebbian accumulated over recursion" control or argument is needed; the comparison is simply against TRM. The tonic same-time kv term is an engineering detail (without it the learned kernel's scale is too small, even with the window area initialised to 1), not a theory point.
Evidence items to show rigorously in the paper (not now): (1) TRM Sudoku checkpoint extrapolation vs ours - corrected 10-30: the only old-repo record (zxcyxv/something commit 83aefb2, runs/sudoku_compare_seg1024_2048, script lt/compare_sudoku.py) is a TRM-MLP run stopped at 32 segments with no step-16 value, so no TRM error-reduction rate exists; 'much smaller for TRM' is unsupported. User decision 10-30: compare only against public TRM checkpoints we run ourselves at matched training depth (the 42x16 retrain) and the same segment-stacking protocol; literature numbers (InfiLoop -38%, Krishna 2026 -43..-69%) are not borrowed; (2) window-shape analysis, one line each: how many heads cancel the fixed-point Hebb gain (bands M0 -0.6..-0.96 in 7/8 heads), sign change by order (Sudoku head 2 causal-potentiating, heads 0 and 3 inverted STDP, others anti-Hebbian); (3) Sudoku bands 10k-block extrapolation -55.3% error, no relapse, 89/92 unsolved wander instead of locking (docs/results.md, claude_2026-10-18_handoff.md).

**Second thesis (Novelty/Universality):** TEM/TEM-t factorization applied to ARC (no prior found; HRM only cites TEM; MapFormer is the closest generalization of TEM-t). Rule head theta = associative memory from demos; frame term = position-key / colour-value retrieval.

**Dropped by the user from Theory:** RoPE 1D finding (just a return to convention), holdout/memorization (already known from HRM..URM), quantitative ARC-2 diagnostics. ARC-2 0% must still be in the main results table (track is ARC-AGI-2).

**Model name (user, 10-30):** TDP-Reasoner (timing-dependent plasticity reasoner). Define it once as STDP applied along the recursion depth, not as a new plasticity concept; explain the absence of spikes as: in this kind of recursive reasoning, spiking itself carries no substantive information, and the window depends only on the relative timing (order and lag) of two events.
**Terminology:** use "temporally asymmetric Hebbian rule with eligibility traces and a learned (STDP-like) kernel", "LTP-like/LTD-like lobes" defined on the association v k^T (activities are signed population codes).

**Gaps:** old numbers need their evidence files re-checked when writing; citations from memory need checking.
Related: [[state-2026-10-27]], [[decisive-experiments]], [[no-multi-seed]].

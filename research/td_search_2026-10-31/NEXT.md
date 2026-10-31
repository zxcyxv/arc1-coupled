# Start here: state of the TD / three-factor work after 2026-10-31

Read this before `README.md` in this folder. The README is the full chronological record, and several of its
earlier conclusions are superseded. This page lists what still holds.

## User rules for this work (non-negotiable)
- **No confidence heuristics.** The user strongly dislikes them: they are only self-consistency, not a method. Never
  use any of these as a method:
  - the lowest-entropy pick;
  - the least confident cell;
  - majority vote.
  Selection and branching must come from learned components trained on outcomes: a value trained by TD, or a policy
  trained on search results.
- **No hand-coded task rules at inference.** No Sudoku conflict checks or verifiers, also inside search.
- **No runs without the user's go.** First state the arm, config, time and comparison. Do not re-run the bands baseline
  (`runs/tf6_hard/bands` exists).
- **Minimal change to the handoff design.** Flag every departure. Explain the architecture before any numbers.
- **Settle a design with RL theory first.** Check the critic-error term, whether the path that is credited is the
  path that was taken (on-policy), and how the trace bridges the reward delay. Run only what the theory leaves open.
- **Explain your own reasoning directly.** Do not stall with literature searches. Single seed, decisive runs.

## What holds (6x6-hard, tiny bands model)
1. **The handoff delta is not circular when it is accumulated on a trace.**
   - Setup: delta_n = -(1-g)C^(h_{n-2}) + gV(h_{n-1}) - V(h_{n-2}), on a decaying trace with tau >= 8.
   - It correlates 0.26-0.83 with the true advantage of the noise drawn in a block (L 16 blocks later).
   - Block by block, the same delta correlates about 0. The "delta is circular" claim in README sections 1 and 6 came
     from that per-block test and is wrong. (`scripts/gate_trace.py`)
2. **Scale-free rule (`tf_unit_scale`):**
   - delta / its batch RMS;
   - eligibility (xi/sigma)^T k~;
   - variance-preserving trace C = lam C + sqrt(1-lam^2) x.
   With these, beta = 1 means a term the size of a write.
3. **Trained, no gain over bands.**

   | arm | exact at 16 blocks (/2048) | [EXTRAP] best (/512) |
   |---|---|---|
   | bands | 1,401 | 421 |
   | `tf_np_unit` (learned beta) | 1,388 | 416 |
   | `tf_np_rule1` (fixed beta = 1, `tf_rule_only`) | 1,388 | 405 |

   - A learned beta stays near 0: backprop sees 4-block segments, while the rule acts 8-16 blocks later. Use a fixed
     rule.
   - With beta = 1 the term is active, at 7-21% of the write. It still does not help.
4. **Why it does not help: exploration.**
   - Per-block v noise has a low ceiling. At sigma 0.1, the best of 16 runs (labels pick) is 1,853 of 2,048.
   - sigma >= 0.3 breaks the solver.
   - A reward rule can only reinforce what exploration reaches. (`scripts/noise_ceiling.py`)
5. **Where the answers are: a persistent, localised change.**
   - Fixing one cell's input to a candidate digit for the whole rollout reaches the answer: the labels-pick ceiling is
     2,013.
   - The working ingredient is that the change is persistent and local, not that it is a discrete token.
   - The reported 2,010 "branch search" result used confidence heuristics to choose the cell and to pick the board. Do
     not cite it as a method.

## Superseded or rejected (do not resume)
- **MCTS V^ in delta, without the root following the search:** off-policy, so it dilutes the signal (section 6).
- **Consolidation (A = sum delta_k (W_k - mean)):**
  - It replaced all three parts of the handoff rule, used entropy for delta, and added an extra memory and gain.
  - The user rejected it. Its 1,968 relied on entropy-scored hypothesis branches.
- **The 8-block lookahead and the 8-slot trace buffer:** unnecessary. The decaying trace with the 1-block delta already
  gives the delayed credit (TD(lambda) backward view).
- **The noise-branch consolidation runs:** stopped and deleted.

## Candidate next steps (none approved yet)
- **Structured exploration plus the fixed rule.** Perturb one cell's input, held for the rollout, and keep
  `tf_np_rule1`'s rule (beta = 1, variance-preserving trace, the 1-block handoff delta). Where to perturb must come
  from a learned policy or at random, never from confidence.
- **TD + MCTS done properly (AlphaZero / TreeStrap).** The root follows the search. A policy proposes the cell and digit,
  and is trained on search outcomes. The value V, trained by TD on true costs, evaluates. The open check: does V learn
  more than confidence? In the gate its branch picks matched entropy, at 87-90%.

## Files
- **Runs:** `runs/tf6_hard/{bands, tf_np_unit, tf_np_rule1, heads}`. `runs/` is gitignored; the eval logs are copied to
  `results/`.
- **Code:**
  - `lt/model.py` flags `three_factor`, `tf_elig_noise`, `tf_unit_scale`, `tf_rule_only`, `tf_beta_init`,
    `tf_elig_tau`;
  - tests `tests/test_lt_three_factor.py`;
  - arms in `scripts/tf_runs.sh`.

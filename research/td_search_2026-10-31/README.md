# Three-factor (TD) learning, V-information and rule-free search on 6x6 Sudoku (2026-10-31)

GPU session record (RTX 5090). Branch `claude/clever-rubin-bw64wm`. Nothing here is committed unless the user asks. Every number
is single seed (project rule). The scripts in `scripts/` were run from the session scratchpad. Paths inside them point there;
adjust before rerunning. The logs are in `results/`.

## Testbed
- **6x6 Sudoku (2x3 boxes):**
  - Generator: `data/make_sudoku6.py`. `--hard` keeps only the puzzles that naked and hidden singles cannot finish.
  - `data/sudoku6_hard_lt.npz`: 1,000 training puzzles and 2,048 test puzzles. Every puzzle needs a branch, and 52% need a backtrack.
  - The easy 6x6 set was 94.7% singles-solvable, so it could not separate the arms.
- **Model:** the tiny bands model, `configs/tf/sudoku6_hard_tiny.json`. Width 64, 4 blocks per segment x 4 segments, noise 0.1.
  - Baseline run: `runs/tf6_hard/bands`. [EXTRAP] best is 421/512 at 30k steps.

## 1. The handoff's three-factor design is circular
The design is delta = -(1-g) C^(s) + g V(s') - V(s), and delta modulates the W writes. There is no real reward anywhere:
delta is one head's prediction minus another's. Measured on puzzles still stuck at the training depth (`scripts/diag*.py`,
`scripts/tf_grader_*.py`, `scripts/tf_credit_np.py`):
- **delta carries no outcome information.** Across noise draws, corr(delta, true outcome) is about 0.
  - delta is dominated by the critic's state residual.
  - Label-free graders all act as confidence meters: C^, V, entropy, and a board-level attention grader.
- **The mechanism itself works with a real reward.** With the oracle reward, node perturbation recovers about 60% of the
  one-step exact-gradient gain.
- **Per-cell credit is worse than board credit.** Scoring one block after the perturbation fails even with labels.
- **Scale issues:**
  - Adam moves beta by about lr per step.
  - The (1-g) normalisation and the (1-eta) leak shrink the term.
  - eta is pinned at its cap.
- **The rule this taught:** an inference-time learning rule needs a real information source at inference
  (`docs/agent_memory` note `check-information-source`).

## 2. ARC gates (`checkpoints/small`, the <= 10 harness)
- **Demo leave-one-out reward is empty under the TRM protocol.** The demos are trained (LOO CE 0.007). Correcting W with it
  does nothing.
- **An unseen held-out demo (holdout model) helps only weakly.** On the rule path it gives cos about 0.3, and one exact-gradient
  step lowers query exact.
- **Per-task lasso lambda selection by the held-out demo fails**, also over 8 augmented versions.
  - The shrinkage confound is real: hard thresholding plus norm preservation makes sparsification neutral.
- **The useful signal is the held-out demo's correctness.** It predicts query correctness: 71% vs 9%, corr +0.70 over versions.
  This is a selection signal.

## 3. V-information of the recursion
Xu et al. 2020, Def. 2-3. Linear probes, 2-fold cross-fitted. Scripts `scripts/vinfo*.py` and `scripts/collapse_pattern.py`.
- **Extrapolation is information gain:** usable information peaks beyond the training depth (block 30).
- **Overthinking in solved puzzles is information loss.** The memory leaves its trained regime: the slow band has e-fold 39
  blocks, longer than the 16-block training depth.
  - Resetting the traces and W every 16 blocks removes the loss: exact goes 0.80 -> 0.85 at block 128.
  - The band period itself is not the culprit (the user's decision: the period drives extrapolation).
- **Lost puzzles collapse from block 5.**
  - Errors propagate, and 32% of the right cells get overwritten.
  - They end as confident boards with swapped digit pairs that keep the clues, so they always contain a conflict (the
    puzzles have unique solutions).
- **Late solvers never collapsed:** I_V was about 1.3 nats before the solve.

## 4. Rule-free search (no task rules anywhere; MuZero-style)
Scripts `scripts/search_*.py`, `scripts/restart_control.py`.
- **Procedure:**
  1. Commit the least confident blank cell to each of its top-3 digits as a hypothetical input clue.
  2. Re-solve each branch from a fresh state for 16 blocks.
  3. Keep the most confident board (lowest entropy), or the board a learned evaluator rates highest.
- **Results** (30k model, 2048 test puzzles):

| method | blocks | exact |
|---|---|---|
| plain recursion | 64 | 1,686 |
| branch search, entropy pick | 64 | 2,010 |
| branch search, label oracle | 64 | 2,013 |
| 4 independent noisy runs, entropy pick / any-run oracle | 64 | 1,597 / 1,611 |

- **The gain comes from the hypotheses, not from resampling.** The errors are systematic.
- **Two levels add little** (2,015).
- **The 16k model shows the same picture:** 1,406 -> 1,896.

## 5. Three-factor consolidation, trained end to end
Script `scripts/consol_train.py` (both arms from scratch, 12k steps, same trainer). The arm `consol` works like this:
1. At block 16, run the branch search above.
2. Compute delta_k = -(entropy_k - mean) / max|.|. It is label-free.
3. Write A = sum_k delta_k (W_k - mean W_k) into a persistent fast memory.
4. The root keeps its own input (no hypothetical clue) and reads W + g_h A for 8 more blocks. g_h is learned from 0, as
   10 x a parameter.

Results (2048 test puzzles, noise off; `results/consol_*_eval.json`):

| | plain arm | consol arm |
|---|---|---|
| plain recursion, 24 blocks | 1,618 | 1,433 |
| consolidation, 72 blocks | (gain stays 0) | **1,968** |
| explicit selection, 64 blocks (reference) | 1,998 | 1,992 |

Mechanism check on the consol arm (`scripts/consol_ablate.py`):

| delta used for A | exact |
|---|---|
| trained delta | 1,968 |
| shuffled across branches | 1,150 |
| negated | 404 |
| A = 0 | 1,433 |
| label-derived delta | 1,970 |

**Conclusion:** the outcome-dependent, sign-changing write carries the search result into the fast memory of a single
trajectory. The label-free third factor is as good as the label one, because the branch search supplies information that
the plain recursion lacks.

## 6. TD + search (minimal change to the handoff rule)
The handoff rule is kept, and only V(h_{n-1}) in delta is replaced, at the first block of each segment (every 4 blocks):

    delta_n = -(1-g) C^(h_{n-2}) + g V^(h_{n-1}) - V(h_{n-2})
    C_n = lam C_{n-1} + (1-lam) S_n,   W~_n = (1-eta) W~_{n-1} + eta S_n + eta beta delta_n C_{n-1}   (unchanged)

**Step 1, heads only** (`scripts/heads_train.py` -> `runs/tf6_hard/heads/heads_30000.pt`, 281 s):
- The body is the frozen `runs/tf6_hard/bands/step_30000.pt`, with beta = 0, so the trajectories are the frozen
  model's own.
- Only the C^ / V heads are trained, with the handoff losses (`LT.tf_finish`):
  - C^ -> the true cell CE;
  - V -> TD, g = 0.96875.
- The rest of the training setup is the same as the original: 30k steps, 128 lanes, 4 x 4 blocks, noise 0.1.
- Losses: c_loss 0.280 -> 0.253, v_loss 0.0064 -> 0.0014. Both plateau by 5k steps.

**Step 2, gate** (`scripts/gate_vhat.py`; 2,048 test puzzles, noise off, boundaries b = 4, 8, 12):
- V^ is an MCTS-style search value:
  - Branches: the least confident blank cell x its top-3 digits, each as a hypothetical input clue.
  - Each branch is a copy of the whole state, rolled out 8 blocks.
  - Branch value = the 8-step return, -(1-g) sum_j g^j C^ + g^8 V(leaf).
  - V^ = the max over the branches.
- Heads: corr(C^, true CE) over puzzles 0.61-0.73; corr(V, -CE 8 blocks later) 0.19-0.43.
- A. Does the chosen branch hold the true digit (puzzles whose true digit is in the top-3)?
  - V^ 86.9 / 88.4 / 89.8%, entropy 87.2 / 88.9 / 90.0%. Passes (bar: within 5 points of entropy).
- B. Median per-puzzle corr(delta, true advantage) over 16 draws of the v noise in block b. The true advantage is
  -(L_d - mean L) with L = the CE 8 blocks later.
  - delta(V): -0.06 / -0.05 / -0.01.
  - delta(V^): +0.23 / +0.26 / +0.20, positive in 69-77% of puzzles.
  - This is between the pre-set stop bar (0.1) and pass bar (0.3).

**Control: hypotheses or lookahead?** (`scripts/gate_ctrl.py`, same draws, seed 0)
- The control replaces V^ by the 8-step return of the root's own 8-block rollout, with no hypothesis.

| boundary | delta(V) | delta(V^) MCTS | delta(max leaf V) | control: lookahead only |
|---|---|---|---|---|
| H = 8: 4 / 8 / 12 | -0.07 / -0.09 / -0.02 | +0.28 / +0.26 / +0.24 | +0.20 / +0.14 / +0.08 | **+0.96 / +0.88 / +0.83** |
| H = 16: 4 / 8 / 12 | -0.05 / -0.05 / -0.01 | +0.22 / +0.15 / +0.11 | +0.17 / +0.12 / +0.04 | **+0.71 / +0.42 / +0.18** |

- The signal comes from the lookahead.
- The hypotheses dilute it:
  - V^'s best branch is the root's top-1 digit only 38-41% of the time, so V^ mostly values paths the root does not take.
  - corr(V^, control) over draws is 0.28-0.35.
- At H = 8 the control's last state is the state the true CE is measured on, so its values are inflated. H = 16 has no
  such overlap.
- A single block's noise effect mostly washes out by 16 blocks: the median std of the advantage is 1e-2 at H = 8 and
  3e-5 to 4e-4 at H = 16.
- **User decision: MCTS dropped.**

**Why the lookahead should work (design reasoning):**
- The rule W += eta beta delta C has the TD(lambda) / actor-critic form "TD error x eligibility". It improves things
  only if delta carries the true advantage.
- The handoff delta = r + gV(s') - V(s) carries no information outside the critic:
  - r = -(1-g)C^ is itself a prediction;
  - one block's write changes h too little for V to see it.
- The n-step target replaces gV(s') by the costs actually incurred over n blocks plus g^n V at the end. The write's
  effect, invisible in h_{n-1}, has become visible on the board by then, and C^ judges a visible board well.
- Standard theory gives the form: delta x eligibility, and the n-step TD target.
- My own choices: R = 8 (from the measurements), every 4 blocks (compute), and a noise-free rollout.
- Not guaranteed: that steering W by the scalar delta on the trace C improves solving. That is the decisive run's
  question.
- At inference no external information enters. The lookahead previews the root's own future, so any gain has to come
  from acting on the preview.

## 7. Delayed-reward variant (user's proposal, not yet implemented)
- The write of block m is rewarded at block m+8, using the 8-step TD error computed on the actual trajectory:

      delta_m(8) = -(1-g) sum_{k=0..7} g^k C^(h_{m-1+k}) + g^8 V(h_{m+7}) - V(h_{m-1})
      W~_{m+8}  += eta beta delta_m(8) C_m      (C_m: the original trace as it was at block m, kept in an 8-slot buffer)

- What it gives:
  - This is online n-step TD: it waits instead of simulating. It is the distal-reward form of three-factor learning
    (Izhikevich 2007).
  - It costs no extra compute and gives a reward every block from block 9 on (8 rewards per 16-block training puzzle).
  - At noise-free inference, delta_m(8) equals the control's delta up to one step, so the control numbers above apply.
- Why the trace is buffered:
  - The original trace (tau = 4) has nearly forgotten block m's write by block m+8: the weight of S_1 in C_8 is 0.033,
    against 0.25 for S_8.
  - So the trace from 8 blocks ago is kept and rewarded.
- Costs:
  - Correction comes after the fact: the 8 blocks have already run. Lost puzzles collapse from block 5.
  - In training, the noise of the blocks in between adds variance to delta.

## 8. Lessons: about half of today's work was waste from not applying basic RL theory
Each item below could have been settled on paper from standard TD theory before running anything.
- **The label-free graders (C^, V, entropy, board-attention grader) as delta.**
  - A TD error carries the true advantage only when the critic is right. Otherwise delta = advantage + critic error.
  - One block's write is too small for a learned critic to see, so the critic-error term had to dominate.
  - Several gate scripts measured this one fact again and again.
- **Consolidation (section 5).**
  - It replaced delta, the eligibility and the write all at once, instead of fixing the one broken part, delta's
    information source.
  - The number was good (1,968), but it answered a different question from the handoff design, and the user rejected it.
- **Multi-cycle consolidation at inference.** It was run out of distribution (summed A, depth past training) with no
  hypothesis worth testing.
- **MCTS V^ (section 6).**
  - A max over hypothesis branches values paths the root does not take. In TD + search (TD-Leaf, AlphaZero), the agent
    follows the search; here only delta changed.
  - So, off-policy, V^ could not judge the root's own writes. I named this risk only after building it, and the
    control confirmed it (the top-1 branch was chosen only 38-41% of the time).
- **The lookahead.**
  - It is simply an n-step return, and online n-step TD (wait n blocks, then reward the buffered trace) gives the same
    information at no compute.
  - The user proposed the delayed version, not I.
  - The right order would have been to start from n-step TD with an eligibility trace bridging the delay (the
    distal-reward setting of Izhikevich 2007), and only then consider search.

What was not wasted: the circularity diagnosis, the gate method (delta vs true advantage over noise draws), the
V-information analysis, and the rule-free branch-search result (1,686 -> 2,010).

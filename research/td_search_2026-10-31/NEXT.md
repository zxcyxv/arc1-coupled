# Next steps (candidates, as of 2026-10-31)
Nothing below is launched without the user's go. State the arm, the config, the time and the comparison first.

## Decisions pending from the user
1. **Which delta for the decisive run:**
   - (a) the delayed reward of README section 7 (the user's proposal; recommended: no extra compute, a reward every
     block);
   - (b) the 8-block lookahead every 4 blocks (section 6).
2. **Trace for (a):** an 8-slot buffer of past traces C_m (recommended; exact "block 1 rewarded at block 9"), or a longer
   trace (tau ~ 8 or more; credit smeared over blocks 1-8).
3. **beta scale:**
   - Under Adam, beta moves by about lr = 1e-4 per step, so at most about 3 in 30k steps. The earlier diagnosis found
     the term is only 0.6-1.6% of W even at beta = +2.
   - Option: `tf_beta_scale=100` in every arm (a departure from the original, applied equally).
4. **(b) only:** the delta of the other 3 blocks of each segment. Keep the handoff delta(V), which carries no signal,
   or set it to 0.

## Decisive run (step 3), once the decisions are made
- **Arms:** the original tf (`three_factor=true`) vs the chosen variant, from scratch.
  - Config `configs/tf/sudoku6_hard_tiny.json` + `tf_noise=0.1` (as `runs/tf6_hard/bands`), 30k steps, single seed.
  - `runs/tf6_hard/bands` is the existing reference; do not re-run it.
- **Metrics:** [EXTRAP] best exact, and exact at 16 blocks.
- **Time (estimate):** about 5 min per arm for (a); about 1.7x that in training and 3x at inference for (b).
- **Implementation:**
  - A flag in `lt/model.py`; the carry needs the buffer of 8 traces and the last 9 (C^, V) board values.
  - Unit test: the flag off is bitwise equal to the original.
  - Tests in `tests/test_lt_three_factor.py`.

## Optional measurements (read-only, seconds; ask first)
- Gate for (a) under training noise: corr(delta_m(8), true advantage) with noise in every block (common across draws
  except block m).
- Effect of the rule on the frozen model: no training; set beta by hand and see whether exact at 16-64 blocks moves.
  This is an eval-time knob and not decisive (memory `decisive-experiments`).

## Other open threads from today
- Rule-free branch search (README section 4) works as explicit selection: 1,686 -> 2,010 / 2,048. It is not part of
  the TD design.
- Overthinking fix: resetting the traces and W every 16 blocks takes exact from 0.80 to 0.85 at 128 blocks. Not
  pursued.
- Three-factor is out of the ARC paper (due 11-08), by user decision.

#!/bin/bash
# A run under a dollar cap, then its results committed and pushed (runs in the foreground; launch it with setsid nohup).
#
#   RUN=arc1_b30g10_rope1d_ffn BUDGET=50 PRICE=6.94 MARGIN_H=0.25 \
#     EXTRA="epochs=5000 final_versions=8 model.blocks=30 model.grad_blocks=10 ..." bash scripts/capped_run.sh
#
# MAX_HOURS = BUDGET / PRICE - MARGIN_H (the margin covers start-up, the final evaluation and the commit). arc.train stops at
# the first lane replacement after MAX_HOURS and writes a checkpoint; it skips the final evaluation once past the deadline.
# Afterwards, finished or not: run.sh archive (config, logs, evaluations -> docs/evidence/run_$RUN; EMA milestones ->
# checkpoints/run_$RUN), the latest full checkpoint split into EMA and raw weights (no optimizer state, under GitHub's 100 MB
# limit), a commit and a push. STOP_INSTANCE=1 then stops this Vast instance (GPU billing ends; the disk is kept).
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:?set RUN}
BUDGET=${BUDGET:?set BUDGET (dollars)}
PRICE=${PRICE:?set PRICE (dollars per hour)}
MARGIN_H=${MARGIN_H:-0.25}
MAX_HOURS=$(python3 -c "print(round($BUDGET / $PRICE - $MARGIN_H, 3))")
export RUN MAX_HOURS
echo "== $(date -u +%FT%TZ) capped run $RUN: budget \$$BUDGET at \$$PRICE/h -> MAX_HOURS $MAX_HOURS"
bash scripts/run.sh start
log=runs/$RUN/stdout.log
until grep -q '^EXITCODE' "$log" 2>/dev/null; do sleep 60; done
code=$(grep '^EXITCODE' "$log" | tail -1)
echo "== $(date -u +%FT%TZ) $RUN exited: $code"
bash scripts/run.sh archive
latest=$(ls -1 runs/$RUN/step_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)
if [ -n "$latest" ]; then
  python - "$latest" "checkpoints/run_$RUN" <<'EOF'
import sys, torch
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
c = torch.load(src, map_location='cpu', weights_only=False)
torch.save(dict(format=c['format'], step=c['step'], model=c['ema'], weights='ema', config=c['config']),
           dst / f"step_{c['step']}_ema.pt")
torch.save(dict(format=c['format'], step=c['step'], model=c['model'], weights='raw', config=c['config']),
           dst / f"step_{c['step']}_raw.pt")
print('saved EMA and raw weights of step', c['step'])
EOF
fi
git add "docs/evidence/run_$RUN"
git add -f checkpoints/run_$RUN/*.pt 2>/dev/null
git commit -q -m "Archive $RUN (capped at \$$BUDGET, $code)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AfmdaZyFqRRPnUqCs3TX7j" && git push -q origin HEAD
echo "== $(date -u +%FT%TZ) committed and pushed: $(git log --oneline -1)"
if [ "${STOP_INSTANCE:-0}" = 1 ]; then
  echo "== stopping instance $CONTAINER_ID"
  vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
fi

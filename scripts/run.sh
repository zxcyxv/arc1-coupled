#!/bin/bash
# ARC-AGI-1 coupled main run (docs/decisions.md) on one multi-GPU instance. From the repository root:
#   bash scripts/run.sh setup     torch 2.11 (cu128, sm_120) + numpy if missing, GPUs, NCCL (P2P fallback), disk,
#                                 data/arc1_trm.npz checksum, unit tests
#   bash scripts/run.sh bench     s/step and peak memory for grad_accum 1, 2, 4 (48 steps each); keeps the fastest that fits
#   bash scripts/run.sh start     launch (or resume) runs/$RUN in the background (setsid nohup torchrun)
#   bash scripts/run.sh status    workers, last log lines, s/step, ETA, evaluations, GPUs (util, memory, temperature,
#                                 power, clocks, throttle reasons), disk
#   bash scripts/run.sh stop      SIGTERM: the run writes a checkpoint at the next lane replacement (<= 16 steps), exits
#   CKPT=runs/arc1/step_N.pt bash scripts/run.sh eval    1001-version vote of a checkpoint, in the background
#   bash scripts/run.sh archive   config, logs and evaluations to docs/evidence/run_$RUN (no checkpoints)
# Environment: RUN (arc1), CONFIG (configs/arc1.json), NGPU (all GPUs), GA (grad_accum; default: the bench result),
#   EXTRA (more KEY=VALUE overrides, e.g. "epochs=30000"), MAX_HOURS, VENV (/venv/main), VERSIONS (eval: all),
#   BENCH_GA (bench: "1 2 4")
# Disk: the venv with torch ~7 GB, this repository ~15 MB, a run ~1 GB (2 checkpoints of ~60 MB, 10 milestones of ~30 MB,
#   train.jsonl ~80 MB, the torch.compile cache in /tmp ~1 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-/venv/main}
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
mkdir -p runs
[ -f runs/env.sh ] && . runs/env.sh                       # NCCL_P2P_DISABLE=1 when setup found P2P broken
RUN=${RUN:-arc1}
CONFIG=${CONFIG:-configs/arc1.json}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}

worker_pids() {  # the arc.train python processes of runs/$RUN (not torchrun, not the wrapper shell)
  ps -eo pid=,args= | awk -v o="--out runs/$RUN" 'index($0, o) && index($0, "-m arc.train") && !index($0, "torchrun") && !index($0, "bash -c") {print $1}'
}

setup() {
  echo "== repo $(git rev-parse --short HEAD 2>/dev/null || echo '?'), $(git status --porcelain 2>/dev/null | wc -l) uncommitted paths"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit,ecc.mode.current --format=csv,noheader
  nvidia-smi topo -m 2>/dev/null | head -12 || true
  if ! python -c "import sys, torch; sys.exit(0 if torch.__version__.startswith('2.11.') else 1)" 2>/dev/null; then
    # CUDA 12.x wheels run on any driver for CUDA >= 12.0 (minor-version compatibility); Blackwell (sm_120) needs cu128,
    # an older driver (e.g. max CUDA 12.4 on H100 hosts) takes cu126, the build TRM pins (torch 2.7.0+cu126).
    local drv wheel=cu128
    drv=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')
    if awk -v v="$drv" 'BEGIN { split(v, a, "."); exit !((a[1] + 0) * 100 + a[2] < 1208) }'; then wheel=cu126; fi
    echo "== installing torch 2.11.0+$wheel (driver supports CUDA $drv)"
    uv pip install --python "$(command -v python)" "torch==2.11.0" --index-url "https://download.pytorch.org/whl/$wheel"
  fi
  python -c "import numpy" 2>/dev/null || uv pip install --python "$(command -v python)" numpy
  python - <<'EOF'
import torch
print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPUs', torch.cuda.device_count(), 'archs', torch.cuda.get_arch_list())
for i in range(torch.cuda.device_count()):
    cc = torch.cuda.get_device_capability(i)
    if f'sm_{cc[0]}{cc[1]}' not in torch.cuda.get_arch_list():
        raise SystemExit(f'GPU {i} is sm_{cc[0]}{cc[1]}, not in this torch build (need a cu128+ wheel)')
    a = torch.randn(1024, 1024, device=f'cuda:{i}')
    f = torch.compile(lambda x: (x @ x).float().sum())
    assert torch.isfinite(f(a)).item()
print('matmul + torch.compile ok on every GPU')
EOF
  if ! timeout 120 torchrun --standalone --nproc_per_node="$NGPU" scripts/nccl_check.py; then
    echo "NCCL check failed or hung; retrying with NCCL_P2P_DISABLE=1 (the all-reduce is ~15 MB per step, P2P is not needed)"
    NCCL_P2P_DISABLE=1 timeout 120 torchrun --standalone --nproc_per_node="$NGPU" scripts/nccl_check.py
    echo 'export NCCL_P2P_DISABLE=1' > runs/env.sh
  fi
  local free; free=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
  echo "disk: ${free} GB free"
  [ "$free" -ge 5 ] || { echo "need >= 5 GB free"; exit 1; }
  for f in data/*_trm.npz.sha256; do sha256sum -c "$f"; done
  python -m unittest tests.test_model tests.test_data tests.test_evaluate tests.test_train 2>&1 | tail -3
}

bench() {
  mkdir -p runs/bench
  echo "== bench: $NGPU GPUs, $CONFIG, 48 steps per grad_accum (the first ~32 include torch.compile)"
  for ga in ${BENCH_GA:-1 2 4}; do
    if timeout "${BENCH_TIMEOUT:-1500}" torchrun --standalone --nproc_per_node="$NGPU" -m arc.train --config "$CONFIG" \
        --out runs/bench/ga$ga --bench 48 --set grad_accum=$ga ${EXTRA:-} > runs/bench/ga$ga.log 2>&1; then
      grep '^BENCH' runs/bench/ga$ga.log | tail -1
    else
      echo "  grad_accum $ga: failed ($(grep -m1 -oE 'OutOfMemoryError|out of memory|[A-Za-z]*Error[^\n]{0,80}' runs/bench/ga$ga.log || echo timeout))"
    fi
  done
  python - "$CONFIG" <<'EOF'
import glob, json, sys
from arc.data import Store
from arc.train import load_config, mean_puzzle_examples
rows = [json.loads(l.split(' ', 1)[1]) for f in sorted(glob.glob('runs/bench/ga*.log')) for l in open(f) if l.startswith('BENCH ')]
if not rows:
    raise SystemExit('no setting fits')
best = min(rows, key=lambda r: r['s_per_step'])
cfg = load_config(sys.argv[1])
total = int(cfg['epochs'] * 960 * mean_puzzle_examples(Store(cfg['data'])) / cfg['global_batch_size'])
json.dump(best, open('runs/bench.json', 'w'))
print(f"best: grad_accum {best['grad_accum']} ({best['lanes']} lanes x {best['ranks']} GPUs), {best['s_per_step']:.3f} s/step, "
      f"peak {best['peak_gib']:.1f} GiB -> {total:,} steps = {total * best['s_per_step'] / 86400:.2f} days (+ evaluations)")
EOF
}

start() {
  local ga=${GA:-$(python -c "import json; print(json.load(open('runs/bench.json'))['grad_accum'])" 2>/dev/null || echo 2)}
  [ -z "$(worker_pids)" ] || { echo "runs/$RUN is already running: $(worker_pids | tr '\n' ' ')"; exit 1; }
  mkdir -p "runs/$RUN"
  local limits=""
  [ -n "${MAX_HOURS:-}" ] && limits="max_hours=$MAX_HOURS"
  echo "== $(date -u +%FT%TZ) start runs/$RUN: $CONFIG, $NGPU GPUs, grad_accum $ga $limits ${EXTRA:-}" | tee -a "runs/$RUN/stdout.log"
  setsid nohup bash -c "torchrun --standalone --nproc_per_node=$NGPU -m arc.train --config $CONFIG --out runs/$RUN \
      --set grad_accum=$ga $limits ${EXTRA:-}; echo EXITCODE \$?" >> "runs/$RUN/stdout.log" 2>&1 < /dev/null &
  sleep 10
  echo "workers: $(worker_pids | tr '\n' ' ') (log runs/$RUN/stdout.log; bash scripts/run.sh status)"
}

status() {
  echo "workers: $(worker_pids | tr '\n' ' ')"
  [ -f "runs/$RUN/stdout.log" ] && { grep -E '^\[(TRAIN|EVAL)\]|^EXITCODE|Error' "runs/$RUN/stdout.log" | tail -n 6 || true; }
  python - "runs/$RUN" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
cfg = json.loads((d / 'config.json').read_text()) if (d / 'config.json').exists() else None
if (d / 'train.jsonl').exists() and cfg:
    with open(d / 'train.jsonl') as f:
        rows = [json.loads(l) for l in f.readlines()[-400:]]
    if rows:
        sps = sum(r['sec'] for r in rows[-200:]) / len(rows[-200:])
        last = rows[-1]
        ex = [r for r in rows if 'exact' in r]
        print(f"step {last['step']:,}/{cfg['total_steps']:,} (epoch {last['epoch']:.0f}): {sps:.3f} s/step, "
              f"ETA {(cfg['total_steps'] - last['step']) * sps / 3600:.1f} h, loss {last['loss']:.4f}"
              + (f", train exact {ex[-1]['exact']:.4f}" if ex else ''))
if (d / 'eval.jsonl').exists():
    for l in open(d / 'eval.jsonl').readlines()[-4:]:
        m = json.loads(l)
        print(f"eval {m['kind']} step {m['step']:,} epoch {m['epoch']:.0f}: identity {m['identity_exact']}/{m['identity_n']} "
              f"pass@1 {m['pass@1']:.4f} pass@2 {m['pass@2']:.4f} ({m['seconds']:.0f}s)")
EOF
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,clocks_event_reasons.active \
    --format=csv,noheader 2>/dev/null || nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader
  df -h . | tail -1
  du -sh "runs/$RUN" 2>/dev/null || true
}

stop() {
  local p; p=$(worker_pids)
  [ -n "$p" ] || { echo "runs/$RUN is not running"; exit 0; }
  kill -TERM $p
  echo "SIGTERM sent to $(echo $p | tr '\n' ' '): checkpoint at the next lane replacement, then exit (tail -f runs/$RUN/stdout.log)"
}

evaluate() {
  local ck=${CKPT:?set CKPT=runs/<run>/step_N.pt} v=${VERSIONS:-all}
  local log="runs/$RUN/eval_$(basename "$ck" .pt)_$v.log"
  setsid nohup bash -c "torchrun --standalone --nproc_per_node=$NGPU -m arc.evaluate --ckpt $ck --versions $v; echo EXITCODE \$?" \
    > "$log" 2>&1 < /dev/null &
  echo "evaluation running in the background: $log"
}

archive() {  # logs and evaluations -> docs/evidence/run_$RUN; EMA milestones (~15 MB each) -> checkpoints/run_$RUN
  local e="docs/evidence/run_$RUN" c="checkpoints/run_$RUN"
  mkdir -p "$e" "$c"
  cp "runs/$RUN/config.json" "runs/$RUN/stdout.log" "$e/"
  [ -f "runs/$RUN/eval.jsonl" ] && cp "runs/$RUN/eval.jsonl" "$e/"
  [ -f "runs/$RUN/train.jsonl" ] && gzip -c "runs/$RUN/train.jsonl" > "$e/train.jsonl.gz"
  cp runs/$RUN/eval_*.json runs/$RUN/submission_*.json runs/$RUN/milestones/eval_*.json "$e/" 2>/dev/null || true
  cp -u runs/$RUN/milestones/step_*.pt "$c/" 2>/dev/null || true
  du -sh "$e" "$c"
  echo "archived; to push (checkpoints are gitignored, so -f):"
  echo "  git add $e && git add -f $c/*.pt && git commit -m 'Archive $RUN' && git push"
  echo "the latest full checkpoint (optimizer included, ~60 MB) stays in runs/$RUN; push it with git add -f too if the run"
  echo "may have to resume on another instance"
}

cmd=${1:-}; shift || true
case $cmd in
  setup|bench|start|status|stop|archive) $cmd ;;
  eval) evaluate ;;
  *) sed -n '2,18p' "$0"; exit 1 ;;
esac

#!/bin/bash
# Three-factor comparison (lt/, Sudoku-Extreme 1k, configs/tf/base.json) on one instance, one arm per GPU. From the
# repository root:
#   bash scripts/tf_runs.sh setup           torch + numpy if missing, GPUs, data checksum, the lt three-factor unit tests
#   bash scripts/tf_runs.sh start [ARM...]  launch (or resume) runs/tf/$ARM in the background (setsid nohup); the k-th
#                                           arm of the list on GPU k (default list: bands tf_zero tf_w0)
#   bash scripts/tf_runs.sh status          per arm: worker PID, last log line, last [EVAL] / [EXTRAP] / [TF]; GPUs; disk
#   bash scripts/tf_runs.sh stop [ARM...]   SIGTERM to the arm's python worker: it checkpoints and exits
# Arms (only these flags differ): bands = the bands EMA model; tf_zero = three_factor (W from 0 every puzzle);
#   tf_w0 = three_factor + w0_learned (W from a learned W0 every puzzle); w0 = w0_learned alone (control for W0);
#   tf_np = three_factor + node-perturbation eligibility (needs tf_noise > 0 in EXTRA, which every arm then shares);
#   tf_np_s100 = tf_np with beta = 100 x its parameter (Adam moves it 100x faster), from 0; tf_np_b2 = the same from beta 2;
#   tf_np_fix_p2 / _m2 = tf_np with the term detached (tf_rule_only), beta fixed at +2 / -2.
#   tf_np_unit = tf_np with the scale-free third factor (tf_unit_scale: delta / its batch RMS, eligibility (xi/sigma)^T k~),
#   trace tau 16 at the start, beta = 10 x its parameter (from 0).
#   tf_np_rule1 = scale-free third factor as a fixed plasticity rule: beta = 1, not trained (tf_rule_only; the trace's
#   lam is then not trained either, tau 16). (tf_np_unit ran before the trace became variance-preserving.)
# Environment: CONFIG (configs/tf/base.json), EXTRA (more KEY=VALUE overrides for every arm), GPUS (e.g. "0 1 2"),
#   VENV (/venv/main), RUNS (output root, runs/tf). Rerunning start on a stopped arm resumes it (same --out); a changed config is refused.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-/venv/main}
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
CONFIG=${CONFIG:-configs/tf/base.json}
ARMS_DEFAULT="bands tf_zero tf_w0"
RUNS=${RUNS:-runs/tf}

flags() {
  case "$1" in
    bands)   echo "" ;;
    tf_zero) echo "three_factor=true" ;;
    tf_w0)   echo "three_factor=true w0_learned=true" ;;
    w0)      echo "w0_learned=true" ;;
    tf_np)   echo "three_factor=true tf_elig_noise=true" ;;
    tf_np_s100) echo "three_factor=true tf_elig_noise=true tf_beta_scale=100" ;;
    tf_np_b2)   echo "three_factor=true tf_elig_noise=true tf_beta_scale=100 tf_beta_init=2" ;;
    tf_np_fix_p2) echo "three_factor=true tf_elig_noise=true tf_beta_init=2 tf_rule_only=true" ;;
    tf_np_fix_m2) echo "three_factor=true tf_elig_noise=true tf_beta_init=-2 tf_rule_only=true" ;;
    tf_np_unit) echo "three_factor=true tf_elig_noise=true tf_unit_scale=true tf_elig_tau=16 tf_beta_scale=10" ;;
    tf_np_rule1) echo "three_factor=true tf_elig_noise=true tf_unit_scale=true tf_elig_tau=16 tf_beta_init=1 tf_rule_only=true" ;;
    *) echo "unknown arm $1 (bands, tf_zero, tf_w0, w0, tf_np, tf_np_s100, tf_np_b2, tf_np_fix_p2, tf_np_fix_m2, tf_np_unit, tf_np_rule1)" >&2; return 1 ;;
  esac
}

worker_pids() {  # the lt.train python process of runs/tf/$1 (not the wrapper shell)
  ps -eo pid=,args= | awk -v o="--out $RUNS/$1" 'index($0, o) && index($0, "-m lt.train") && !index($0, "bash -c") {print $1}'
}

setup() {
  echo "== repo $(git rev-parse --short HEAD 2>/dev/null || echo '?'), $(git status --porcelain 2>/dev/null | wc -l) uncommitted paths"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    local drv wheel=cu128
    drv=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')
    if awk -v v="$drv" 'BEGIN { split(v, a, "."); exit !((a[1] + 0) * 100 + a[2] < 1208) }'; then wheel=cu126; fi
    echo "== installing torch 2.11.0+$wheel (driver supports CUDA $drv)"
    pip install "torch==2.11.0" --index-url "https://download.pytorch.org/whl/$wheel"
  fi
  python -c "import numpy" 2>/dev/null || pip install numpy
  python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPUs', torch.cuda.device_count())"
  sha256sum -c data/sudoku_lt_1k.npz.sha256
  python -m unittest tests.test_lt_three_factor
  df -h . | tail -1
}

start() {
  local arms=("$@") gpus i=0
  [ ${#arms[@]} -eq 0 ] && read -r -a arms <<< "$ARMS_DEFAULT"
  read -r -a gpus <<< "${GPUS:-$(seq -s ' ' 0 $(( $(nvidia-smi -L | wc -l) - 1 )))}"
  [ ${#arms[@]} -le ${#gpus[@]} ] || { echo "${#arms[@]} arms but ${#gpus[@]} GPUs (set GPUS or name fewer arms)"; exit 1; }
  mkdir -p "$RUNS"
  for arm in "${arms[@]}"; do
    local f out="$RUNS/$arm"
    f=$(flags "$arm")
    if [ -n "$(worker_pids "$arm")" ]; then echo "$arm already running (PID $(worker_pids "$arm"))"; i=$((i + 1)); continue; fi
    mkdir -p "$out"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${gpus[$i]} setsid nohup bash -c \
      "exec python -m lt.train --config $CONFIG --out $out --set $f ${EXTRA:-}" >> "$out/stdout.log" 2>&1 < /dev/null &
    sleep 2
    echo "$arm on GPU ${gpus[$i]}: PID $(worker_pids "$arm" || true), log $out/stdout.log"
    i=$((i + 1))
  done
}

status() {
  for d in "$RUNS"/*/; do
    local arm
    arm=$(basename "$d")
    echo "== $arm: worker $(worker_pids "$arm" | tr '\n' ' ')"
    [ -f "$d/stdout.log" ] || continue
    grep -E '^\[LT\] step' "$d/stdout.log" | tail -1 || true
    grep -E '^\[(EVAL|TF|EXTRAP|EXTRAP-TF)\]' "$d/stdout.log" | tail -4 || true
    grep -iE 'error|traceback' "$d/stdout.log" | tail -3 || true
  done
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader || true
  df -h . | tail -1
}

stop() {
  local arms=("$@")
  [ ${#arms[@]} -eq 0 ] && read -r -a arms <<< "$ARMS_DEFAULT"
  for arm in "${arms[@]}"; do
    local pid
    pid=$(worker_pids "$arm")
    if [ -n "$pid" ]; then kill -TERM $pid; echo "$arm: SIGTERM to $pid"; else echo "$arm: not running"; fi
  done
}

cmd=${1:-status}
shift || true
case "$cmd" in
  setup) setup ;;
  start) start "$@" ;;
  status) status ;;
  stop) stop "$@" ;;
  *) echo "usage: $0 setup | start [ARM...] | status | stop [ARM...]"; exit 1 ;;
esac

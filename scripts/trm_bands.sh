#!/bin/bash
# The bands perception with TRM's puzzle embedding, trained in TRM's harness (trm/pretrain.py, arch=bands), on our
# compact store (data/<benchmark>_trm.npz). Everything the run writes (checkpoints, metrics, logs, the evaluator's
# metadata) goes under the repository's runs/ on disk.
#
#   bash scripts/trm_bands.sh setup                 # once: the TRM python environment (.venv-trm)
#   bash scripts/trm_bands.sh start arc2 [OUT]      # OUT defaults to runs/<benchmark>_bands
#   bash scripts/trm_bands.sh status arc2 [OUT]
#   bash scripts/trm_bands.sh stop arc2 [OUT]
#
# Settings (docs/decisions.md, 10-27): global batch 768 on 4 GPUs (grad_accum 2), lr 1e-4 after 2,000 warmup steps,
# weight decay 0.1 on matrices only, puzzle embedding sign-SGD lr 1e-2 wd 0.1, EMA 0.999, 100k epochs, evaluation every
# EVAL_INTERVAL epochs (default 10,000, TRM's) on EVAL_VERSIONS versions (default all, TRM's 1001-version vote), the
# newest checkpoint only (checkpoint_keep 1).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=${TRM_VENV:-$ROOT/.venv-trm}
cmd=${1:?setup|start|status|stop}

if [ "$cmd" = setup ]; then
    # TRM's requirements without torch (the image's /venv/main torch is inherited) and without adam-atan2 (its CUDA
    # build pulls torch; trm/pretrain.py uses arc.train's port of the same kernel).
    uv venv -q --python /venv/main/bin/python --system-site-packages "$VENV"
    VIRTUAL_ENV=$VENV uv pip install -q einops tqdm coolname pydantic argdantic wandb omegaconf hydra-core numba
    "$VENV/bin/python" -c "import hydra, argdantic, numba, torch; print('ok', torch.__version__)"
    exit 0
fi

bench=${2:?arc1|arc2}
OUT=$(realpath -m "${3:-$ROOT/runs/${bench}_bands}")
DATA=$ROOT/runs/trm_data/$bench
key="run_name=${bench}_bands_$(basename "$OUT")"

case $cmd in
start)
    [ -x "$VENV/bin/python" ] || { echo "no $VENV: bash scripts/trm_bands.sh setup"; exit 1; }
    if pgrep -f -- "$key" > /dev/null; then echo "already running: $key"; exit 1; fi
    if [ ! -f "$DATA/identifiers.json" ]; then
        mkdir -p "$DATA"
        ln -sf "$ROOT/data/${bench}_trm.npz" "$DATA/store.npz"
        "$VENV/bin/python" "$ROOT/trm/make_eval_meta.py" "$DATA"
    fi
    mkdir -p "$OUT"
    ev=${EVAL_VERSIONS:-null}
    cd "$ROOT/trm"
    DISABLE_COMPILE=1 WANDB_MODE=disabled OMP_NUM_THREADS=8 setsid nohup "$VENV/bin/python" -m torch.distributed.run \
        --standalone --nproc-per-node "${NPROC:-4}" pretrain.py arch=bands \
        data_paths="[$DATA]" data_format=store \
        global_batch_size=768 grad_accum=2 lr=1e-4 lr_warmup_steps=2000 weight_decay=0.1 weight_decay_groups=matrices \
        puzzle_emb_lr=1e-2 puzzle_emb_weight_decay=0.1 ema=True \
        epochs=${EPOCHS:-100000} eval_interval=${EVAL_INTERVAL:-10000} eval_versions=$ev \
        checkpoint_every_eval=True checkpoint_keep=1 checkpoint_path="$OUT/ckpt" metrics_log="$OUT/metrics" resume_path="$OUT/resume.pt" \
        "$key" hydra.run.dir="$OUT/hydra" >> "$OUT/stdout.log" 2>&1 < /dev/null &
    echo "started $key -> $OUT"
    ;;
status)
    pgrep -f -- "$key" > /dev/null && echo running || echo "not running"
    tail -c 300 "$OUT/stdout.log" | tr '\r' '\n' | grep -v '^$' | tail -1
    tail -n 2 "$OUT/metrics.eval.jsonl" 2>/dev/null || true
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    df -h "$ROOT" | tail -1
    ;;
stop)
    pids=$(pgrep -f -- "$key" | grep -v "^$$\$" || true)
    [ -n "$pids" ] && kill $pids && echo "stopped $pids" || echo "not running"
    ;;
*) echo "unknown command $cmd"; exit 1 ;;
esac

"""DDP training of the coupled model on ARC-AGI-1 in TRM's recipe (docs/decisions.md).

    torchrun --standalone --nproc_per_node=4 -m arc.train --config configs/arc1.json --out runs/arc1
    python -m arc.train --config configs/arc1.json --out runs/smoke --set max_steps=64 global_batch_size=16

Rerunning with the same --out resumes from its latest checkpoint. Checkpoints are written at lane-replacement
boundaries (every `loops` steps every lane starts a new group), so they hold no carry: weights, EMA, optimizer, step and
the data cursor determine the rest of the run exactly.

One optimizer step = one segment (`blocks` blocks, the last `grad_blocks` with gradients) on every lane of the global
batch, split over ranks and `grad_accum` micro-batches; each micro-batch's carry stays on its GPU. Loss: TRM's lm term
(per lane the mean stablemax CE over its labelled output cells, summed over lanes, divided by the global batch);
gradients all-reduced (SUM). AdamATan2, linear warmup then constant lr, weight decay on every parameter, no clipping
(a non-finite loss or gradient stops the run; the last checkpoint stays). EMA of the parameters for evaluation and
checkpoints. Steps = int(epochs * tasks * mean_puzzle_examples / global_batch_size), TRM pretrain.py's count.

On CUDA with compile, one layer (block + rule head, Coupled.layer) is compiled, and each micro-batch's segment, loss and
backward pass is replayed as a CUDA graph (GraphedMicroBatches; cuda_graphs=false for the eager loop): the same kernels
without the per-kernel launch cost. Gradients live in one flat buffer (one all-reduce, one finiteness check); the
optimizer and the EMA run as _foreach ops (bitwise the per-parameter loop).
"""
import argparse
import json
import math
import os
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer

from .data import Store, TrainStream, default_store_path, fingerprint
from .evaluate import log_stablemax, run as run_eval, version_set
from .model import PAD, Coupled, ModelConfig

FORMAT = 'arc1-coupled-v1'
DEFAULTS = dict(
    data=None,                        # default data/arc1_trm.npz
    model={},                         # ModelConfig fields
    global_batch_size=768, grad_accum=2, loops=16,
    epochs=100000, max_steps=None,    # TRM-nominal epochs; max_steps caps the count
    lr=1e-4, lr_warmup_steps=2000, beta1=0.9, beta2=0.95, weight_decay=0.1, ema_rate=0.999,
    quick_eval_epochs=1000,           # original version only (monitoring)
    vote_eval_epochs=10000,           # vote_versions versions, (votes, demo score) ranking
    vote_versions=8,
    final_eval=True, final_versions='all', eval_lanes=None,   # eval micro-batch lanes (None: the training micro-batch)
    save_every_steps=2048, keep_last=2,
    log_every=50, seed=0, compile=True, max_hours=None,
    cuda_graphs=True,                 # with compile on CUDA: each micro-batch (segment, loss, backward) as a CUDA graph
)
_STOP = False


def load_config(src, overrides=()):
    cfg = json.loads(Path(src).read_text()) if isinstance(src, (str, Path)) else dict(src)
    cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'unknown config keys {sorted(unknown)}')
    cfg = dict(DEFAULTS, **cfg)
    for item in overrides:
        k, v = item.split('=', 1)
        v = json.loads(v)
        if k.startswith('model.'):
            cfg['model'] = dict(cfg['model'], **{k[6:]: v})
        elif k in DEFAULTS:
            cfg[k] = v
        else:
            raise ValueError(f'unknown override {k}')
    cfg['data'] = str(cfg['data'] or default_store_path())
    ModelConfig(**cfg['model'])                                       # validates
    if cfg['save_every_steps'] % cfg['loops']:
        raise ValueError('save_every_steps must be a multiple of loops (checkpoints sit at lane replacements)')
    return cfg


def build_model(cfg, device):
    torch.manual_seed(cfg['seed'])
    return Coupled(ModelConfig(**cfg['model'])).to(device)


def init_distributed():
    """(rank, world size, device); ARC_DEVICE=cpu forces the CPU (gloo) path, e.g. for tests/test_train.py."""
    ws, rank = int(os.environ.get('WORLD_SIZE', '1')), int(os.environ.get('RANK', '0'))
    if torch.cuda.is_available() and os.environ.get('ARC_DEVICE', 'cuda') != 'cpu':
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    if ws > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(minutes=60))
    return rank, ws, device


class AdamATan2(Optimizer):
    """TRM's optimizer, adam-atan2 0.0.3 (csrc/adam_atan2.cu adam_math) in torch ops:
        p *= 1 - lr wd;  m = lerp(m, g, 1 - b1);  v = lerp(v, g^2, 1 - b2)
        p -= (lr / (1 - b1^t)) atan2(m, sqrt(v) / sqrt(1 - b2^t))"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        """The element operations above as _foreach ops over the parameters that share a step count, dtype and device
        (one launch per operation, not per parameter and operation; atan2 has no _foreach form)."""
        for g in self.param_groups:
            b1, b2 = g['betas']
            by_t = {}
            for p in g['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st.update(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                st['step'] += 1
                by_t.setdefault((st['step'], p.dtype, p.device), []).append(p)
            for (t, _, _), ps in by_t.items():
                grads = [p.grad for p in ps]
                ms, vs = [self.state[p]['m'] for p in ps], [self.state[p]['v'] for p in ps]
                torch._foreach_mul_(ps, 1 - g['lr'] * g['weight_decay'])
                torch._foreach_lerp_(ms, grads, 1 - b1)
                torch._foreach_lerp_(vs, torch._foreach_mul(grads, grads), 1 - b2)
                denom = torch._foreach_div(torch._foreach_sqrt(vs), math.sqrt(1 - b2 ** t))
                torch._foreach_add_(ps, [torch.atan2(m, d) for m, d in zip(ms, denom)], alpha=-g['lr'] / (1 - b1 ** t))


def lm_loss(logits, labels, O):
    """TRM ACTLossHead lm term: sum over lanes of the per-lane mean stablemax CE (FP64) over labelled output cells.
    Returns the loss and [exact lanes, lanes, right cells, labelled cells]."""
    y = labels[:, O:].long()
    lab = y != PAD
    cell = -torch.gather(log_stablemax(logits.double()), -1, y[..., None]).squeeze(-1)
    loss = (torch.where(lab, cell, 0).sum(-1) / lab.sum(-1).clamp_min(1)).sum()
    with torch.no_grad():
        right = (logits.argmax(-1) == y) & lab
        stats = torch.stack([((right | ~lab).all(-1)).sum(), torch.full((), len(y), device=y.device), right.sum(),
                             lab.sum()]).double()             # no host-to-device copy (CUDA graph capture)
    return loss, stats


def mean_puzzle_examples(store):
    n, v = store.z['task_train_n'], store.n_versions
    return float((n * v).sum() / v.sum())


def boundary_steps(every_epochs, steps_per_epoch, loops, total):
    """Lane-replacement steps at or after each multiple of every_epochs (TRM-nominal epochs)."""
    if not every_epochs:
        return set()
    out, k = set(), 1
    while True:
        s = int(math.ceil(k * every_epochs * steps_per_epoch / loops)) * loops
        if s > total:
            return out
        out.add(s)
        k += 1


def flat_grads(model):
    """Every parameter's .grad as a view of one zeroed flat buffer (returned), when the parameters share a dtype and
    device (else None: autograd's own .grad tensors). Backward accumulates into the views in place, so the all-reduce
    and the finiteness check each read one tensor, and CUDA graphs see fixed gradient addresses."""
    params = list(model.parameters())
    if len({(p.dtype, p.device) for p in params}) != 1:
        return None
    flat = torch.zeros(sum(p.numel() for p in params), dtype=params[0].dtype, device=params[0].device)
    off = 0
    for p in params:
        p.grad = flat[off:off + p.numel()].view_as(p)
        off += p.numel()
    return flat


def _allreduce_grads(model, ws, flat=None):
    if flat is not None:
        if ws > 1:
            dist.all_reduce(flat)                                 # SUM: the loss is divided by the GLOBAL batch
        return
    params = list(model.parameters())
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    if ws > 1:
        flat = torch.cat([p.grad.reshape(-1) for p in params])
        dist.all_reduce(flat)                                     # SUM: the loss is divided by the GLOBAL batch
        off = 0
        for p in params:
            p.grad.copy_(flat[off:off + p.numel()].view_as(p))
            off += p.numel()


def _any(flag, device, ws):
    """flag: a Python bool, or a bool tensor on device (then no host sync before the all-reduce)."""
    t = flag.reshape(1).int() if torch.is_tensor(flag) else torch.tensor([int(flag)], device=device)
    if ws > 1:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


class GraphedMicroBatches:
    """A step's grad_accum micro-batches as CUDA graphs, one per micro-batch slot: the segment (compiled layers), the
    loss and the backward pass, then the new carry copied into the slot's carry buffers. Batches and carries live in
    static per-slot buffers (load() copies a new plan in, reset() writes initial_carry's fresh-lane values in place),
    gradients accumulate into the fixed .grad views of flat_grads, and the graphs share one memory pool (replayed in
    capture order, never concurrently). Same kernels in the same order as the eager loop; the CPU launches k graphs per
    step instead of every kernel. The first capture follows one eager warm-up micro-batch on the capture stream
    (compilation, autotuning, cuDNN plans, cuBLAS workspaces) whose results are discarded. release() frees the graphs
    and their pool (before an evaluation, which then gets that memory); the next run() captures again."""
    def __init__(self, model, k, gbs, flat):
        self.model, self.k, self.gbs, self.flat = model, k, gbs, flat
        self.batches = self.carries = self.graphs = self.outs = None
        self.stream = torch.cuda.Stream()
        self.warm = False

    def load(self, batches):
        """A new plan's micro-batches (every lane fresh)."""
        if self.batches is None:
            self.batches = [{n: v.clone() for n, v in b.items()} for b in batches]
            x = self.batches[0]['inputs']
            self.carries = [self.model.initial_carry(x.shape[0], x.device) for _ in range(self.k)]
            return
        for sb, b in zip(self.batches, batches):
            for n, v in b.items():
                if sb[n].shape != v.shape or sb[n].dtype != v.dtype:
                    raise ValueError(f'micro-batch {n}: shape or dtype changed under CUDA graphs')
                sb[n].copy_(v)
        self.reset()

    @torch.no_grad()
    def reset(self):
        """Fresh lanes in every slot, with initial_carry's values."""
        for c in self.carries:
            c['h'].copy_(self.model.init_hidden.expand_as(c['h']))
            for n, v in c.items():
                if n != 'h' and v is not None:
                    v.zero_()

    def _micro(self, i, write_carry):
        c = self.model.cfg
        logits, new = self.model.segment(self.carries[i], self.batches[i])
        loss, st = lm_loss(logits, self.batches[i]['labels'], self.model.O)
        # With no-grad blocks first, the backward pass never reads the incoming carry, so the new carry is written
        # (or, in the warm-up, dropped) before it: its blocks are free for the backward pass, as eager frees the old
        # carry. Else after the backward pass.
        early = c.blocks > c.grad_blocks
        if early:
            if write_carry:
                self._write(i, new)
            new = None
        (loss / self.gbs).backward()
        if write_carry and not early:
            self._write(i, new)
        return loss.detach(), st

    @torch.no_grad()
    def _write(self, i, new):
        for n, v in new.items():
            if v is not None:
                self.carries[i][n].copy_(v)

    def capture(self):
        s = self.stream
        s.wait_stream(torch.cuda.current_stream())
        if not self.warm:
            with torch.cuda.stream(s):
                self._micro(0, False)
            self.warm = True
        torch.cuda.synchronize()
        self.flat.zero_()
        torch.cuda.empty_cache()                               # the warm-up's cached blocks, before the pool grows
        pool = torch.cuda.graph_pool_handle()
        self.graphs, self.outs = [], []
        for i in range(self.k):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, stream=s):
                self.outs.append(self._micro(i, True))
            self.graphs.append(g)
        torch.cuda.synchronize()

    def run(self):
        """The step's micro-batches: [(loss, stats)] per slot, valid until the next run()."""
        if self.graphs is None:
            self.capture()
        for g in self.graphs:
            g.replay()
        return self.outs

    def release(self):
        if self.graphs is not None:
            torch.cuda.synchronize()
            self.graphs = self.outs = None
            torch.cuda.empty_cache()


def _signal(signum, frame):
    global _STOP
    _STOP = True


def latest_checkpoint(out):
    cks = sorted(Path(out).glob('step_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
    return cks[-1] if cks else None


def save_checkpoint(out, state, model, opt, ema, cfg, data_sha, keep_last, rank, ws):
    if rank == 0:
        path = Path(out) / f"step_{state['step']}.pt"
        tmp = path.with_suffix('.tmp')
        torch.save(dict(format=FORMAT, step=state['step'], cursor=state['cursor'], model=model.state_dict(), ema=ema,
                        opt=opt.state_dict(), config=cfg, data_sha=data_sha), tmp)
        os.replace(tmp, path)
        for old in sorted(Path(out).glob('step_*.pt'), key=lambda p: int(p.stem.split('_')[1]))[:-keep_last]:
            old.unlink()
    if ws > 1:
        dist.barrier()


def save_milestone(out, state, model, ema, cfg, rank):
    """EMA weights only (~15 MB): what evaluation reads (arc.evaluate --ckpt milestones/step_N.pt)."""
    if rank == 0:
        d = Path(out) / 'milestones'
        d.mkdir(exist_ok=True)
        sd = {k: (ema[k].detach().clone() if k in ema else v) for k, v in model.state_dict().items()}
        torch.save(dict(format=FORMAT, step=state['step'], model=sd, weights='ema', config=cfg),
                   d / f"step_{state['step']}.pt")


class EMAWeights:
    """Swap the EMA parameters in for evaluation."""
    def __init__(self, model, ema):
        self.model, self.ema = model, ema

    def __enter__(self):
        self.live = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.copy_(self.ema[n])

    def __exit__(self, *exc):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.copy_(self.live[n])


def main(argv=None):
    global _STOP
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='JSON values; model.X for model fields')
    ap.add_argument('--bench', type=int, default=0, help='run this many steps (no checkpoints, no evaluation)')
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    rank, ws, device = init_distributed()
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _signal)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (lambda m: print(m, flush=True)) if rank == 0 else (lambda m: None)

    store = Store(cfg['data'])
    data_sha = fingerprint(cfg['data'])
    gbs, k, loops = cfg['global_batch_size'], cfg['grad_accum'], cfg['loops']
    if gbs % (ws * k):
        raise ValueError('global_batch_size must be divisible by ranks * grad_accum')
    lanes = gbs // (ws * k)
    mpe = mean_puzzle_examples(store)
    spe = store.n_tasks * mpe / gbs                                     # steps per TRM-nominal epoch
    total = int(cfg['epochs'] * store.n_tasks * mpe / gbs)
    if cfg['max_steps']:
        total = min(total, int(cfg['max_steps']))
    if args.bench:
        total = args.bench

    model = build_model(cfg, device)
    if ws > 1:
        with torch.no_grad():
            for t in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(t, src=0)
    opt = AdamATan2(model.parameters(), lr=0.0, betas=(cfg['beta1'], cfg['beta2']), weight_decay=cfg['weight_decay'])
    ema = {n: p.detach().clone() for n, p in model.named_parameters()}
    state = dict(step=0, cursor=0)
    ck = None if args.bench else latest_checkpoint(out)
    if ck is not None:
        d = torch.load(ck, map_location=device, weights_only=False)
        if d['config']['model'] != cfg['model'] or d['data_sha'] != data_sha:
            raise ValueError(f'{ck} was written with another model config or data file')
        model.load_state_dict(d['model'])
        opt.load_state_dict(d['opt'])
        ema = {n: t.to(device) for n, t in d['ema'].items()}
        state = dict(step=d['step'], cursor=d['cursor'])
        del d
    if rank == 0 and not args.bench:
        (out / 'config.json').write_text(json.dumps(dict(cfg, format=FORMAT, data_sha=data_sha, total_steps=total,
                                                         steps_per_epoch=spe, ranks=ws, lanes=lanes), indent=1) + '\n')
    if cfg['compile'] and device.type == 'cuda':
        import torch._dynamo.config as dc
        import torch._inductor.config as ic
        ic.triton.persistent_reductions = False
        dc.recompile_limit = max(dc.recompile_limit, 16)
        model.layer_fn = torch.compile(model.layer, dynamic=False)        # one block and its rule head
    stream = TrainStream(store, cfg['seed'], lanes, ws * k)
    quick = boundary_steps(cfg['quick_eval_epochs'], spe, loops, total)
    vote = boundary_steps(cfg['vote_eval_epochs'], spe, loops, total)
    eval_lanes = cfg['eval_lanes'] or lanes
    n_par = sum(p.numel() for p in model.parameters())
    log(f"[TRAIN] {FORMAT} torch {torch.__version__} ranks {ws} device {device} params {n_par:,} lanes/micro-batch "
        f"{lanes} x grad_accum {k}; steps {total:,} ({cfg['epochs']} epochs x {store.n_tasks} tasks x {mpe:.4f} / {gbs}); "
        f"{'RESUME ' + str(ck) + ' step ' + str(state['step']) if ck else 'NEW RUN'}; model {cfg['model']}")

    deadline = float('inf') if cfg['max_hours'] is None else time.monotonic() + 3600 * cfg['max_hours']
    began = time.monotonic()
    if ck is not None and state['step'] >= total:                       # a finished run restarted: nothing to do
        log(f"[TRAIN] {ck} is already at step {state['step']} >= {total}; raise epochs/max_steps to continue, "
            f"or evaluate it with scripts/run.sh eval")
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    carries, batches = [None] * k, [None] * k
    flat = flat_grads(model)
    graphed = None
    if cfg['cuda_graphs'] and cfg['compile'] and device.type == 'cuda' and flat is not None:
        graphed = GraphedMicroBatches(model, k, gbs, flat)
    train_log = (out / 'train.jsonl').open('a') if rank == 0 and not args.bench else None
    window, stopped = [], False
    last_exact = None
    bench_t = []

    def evaluate(kind, versions):
        t0 = time.monotonic()
        if graphed is not None:
            graphed.release()                                          # the graphs' pool goes to the evaluation
        with EMAWeights(model, ema):
            res = run_eval(model, store, versions, eval_lanes, loops, rank, ws, device, log)
        if rank == 0:
            m = dict(kind=kind, step=state['step'], epoch=state['step'] / spe, seconds=time.monotonic() - t0, **res[0])
            with (out / 'eval.jsonl').open('a') as f:
                f.write(json.dumps(m) + '\n')
            log(f"[EVAL] {kind} step {state['step']} epoch {m['epoch']:.0f}: identity {m['identity_exact']}/"
                f"{m['identity_n']} task-weighted {m['identity_exact_task']:.4f} view {m['view_exact']:.4f} "
                f"pass@1 {m['pass@1']:.4f} pass@2 {m['pass@2']:.4f} cell {m['identity_cell']:.4f} "
                f"cell_eos {m['identity_cell_eos']:.4f} ({m['seconds']:.0f}s)")
            return res

    try:
        while state['step'] < total:
            step = state['step']
            if step % loops == 0:                                       # lane replacement: stop, save, evaluate here
                carries, batches = [None] * k, [None] * k              # every lane is done: free their memory first
                stopped = _any(_STOP or time.monotonic() > deadline, device, ws)
                if not args.bench and step and (stopped or step % cfg['save_every_steps'] == 0):
                    save_checkpoint(out, state, model, opt, ema, cfg, data_sha, cfg['keep_last'], rank, ws)
                if stopped:
                    log(f'[TRAIN] stop requested: checkpoint step {step} written')
                    break
                if not args.bench and step in vote:
                    save_milestone(out, state, model, ema, cfg, rank)
                    evaluate('vote', version_set(store, cfg['vote_versions']))
                elif not args.bench and step in quick:
                    evaluate('quick', [0])
                plan, state['cursor'] = stream.plan(state['cursor'])
                batches = [{n: v.to(device, non_blocking=True) for n, v in stream.micro_batch(g).items()}
                           for g in plan[rank * k:(rank + 1) * k]]
                carries = [None] * k
                if graphed is not None:
                    graphed.load(batches)                               # fresh lanes in the graphs' static buffers
                    batches = None
            t0 = time.monotonic()
            for g in opt.param_groups:
                g['lr'] = cfg['lr'] * min(1.0, (step + 1) / max(1, cfg['lr_warmup_steps']))
            stats = torch.zeros(4, device=device, dtype=torch.float64)
            loss_sum = torch.zeros((), device=device, dtype=torch.float64)
            if graphed is not None:
                for loss, st in graphed.run():
                    loss_sum += loss
                    stats += st
            else:
                for i in range(k):
                    logits, carries[i] = model.segment(carries[i], batches[i])
                    loss, st = lm_loss(logits, batches[i]['labels'], model.O)
                    (loss / gbs).backward()
                    loss_sum += loss.detach()
                    stats += st
            _allreduce_grads(model, ws, flat)
            if flat is not None:                                        # one host sync
                bad = ~(torch.isfinite(loss_sum) & torch.isfinite(flat).all())
            else:
                bad = not (torch.isfinite(loss_sum) and all(torch.isfinite(p.grad).all() for p in model.parameters()))
            if _any(bad, device, ws):
                raise FloatingPointError(f'non-finite loss or gradient at step {step}; the last checkpoint is intact')
            opt.step()
            if flat is not None:
                flat.zero_()                                            # the .grad views stay (fixed addresses)
            else:
                opt.zero_grad(set_to_none=True)
            model.block.project_eta_(model.cfg)
            with torch.no_grad():
                names = [n for n, _ in model.named_parameters()]
                ev = [ema[n] for n in names]
                torch._foreach_mul_(ev, cfg['ema_rate'])
                torch._foreach_add_(ev, [p.detach() for _, p in model.named_parameters()], alpha=1 - cfg['ema_rate'])
            red = torch.cat([loss_sum[None], stats])
            if ws > 1:
                dist.all_reduce(red)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            sec = time.monotonic() - t0
            state['step'] = step + 1
            seg = step % loops + 1
            if rank == 0:
                loss_v, ex, n, right, cells = red.tolist()
                rec = dict(step=step + 1, seg=seg, loss=loss_v / gbs, lr=opt.param_groups[0]['lr'], sec=round(sec, 4),
                           epoch=round((step + 1) / spe, 3), data_epoch=round(state['cursor'] / store.n_tasks, 3))
                if seg == loops:
                    rec.update(exact=ex / n, cell=right / max(cells, 1))
                    last_exact = rec['exact']
                if train_log:
                    train_log.write(json.dumps(rec) + '\n')
                    if seg == loops:
                        train_log.flush()
                window.append(sec)
                bench_t.append(sec)
                if (step + 1) % cfg['log_every'] == 0:
                    sps = float(np.mean(window[-cfg['log_every']:]))
                    mem = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == 'cuda' else 0.0
                    log(f"[TRAIN] step {step + 1:,}/{total:,} epoch {rec['epoch']:.1f} loss {rec['loss']:.4f} "
                        f"lr {rec['lr']:.2e} {sps:.3f}s/step ETA {(total - step - 1) * sps / 3600:.1f}h "
                        f"mem {mem:.1f}GiB" + (f' train exact {last_exact:.4f}' if last_exact is not None else ''))
                    window = window[-cfg['log_every']:]
        if args.bench:
            if rank == 0:
                warm = min(len(bench_t) // 2, 2 * loops)
                mem = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == 'cuda' else 0.0
                print('BENCH ' + json.dumps(dict(grad_accum=k, lanes=lanes, ranks=ws, steps=len(bench_t),
                                                 s_per_step=float(np.mean(bench_t[warm:])), peak_gib=mem)), flush=True)
            return
        if not stopped and state['step'] >= total:
            save_checkpoint(out, state, model, opt, ema, cfg, data_sha, cfg['keep_last'], rank, ws)
            save_milestone(out, state, model, ema, cfg, rank)
            log(f"[TRAIN] finished at step {state['step']} ({(time.monotonic() - began) / 3600:.1f}h)")
            if cfg['final_eval'] and time.monotonic() < deadline:
                res = evaluate('final', version_set(store, cfg['final_versions']))
                if rank == 0:
                    from .evaluate import submission
                    (out / f"submission_step_{state['step']}.json").write_text(
                        json.dumps(submission(store, res[1], res[2])))
    finally:
        if train_log:
            train_log.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()

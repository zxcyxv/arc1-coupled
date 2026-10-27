"""Small single-GPU harness: the previous repository's ARC-1 <= 10 ablation (zxcyxv/something lt/coupled_train.py and
lt/coupled.py CoupledEpisodes) on this repository's model (arc.model.Coupled, fused kernels, compiled layer, CUDA graph).

    CUDA_VISIBLE_DEVICES=3 python -m arc.small --out runs/small/b8g4 --set model.blocks=8 model.grad_blocks=4

Data: data/arc1_ctx10_trans.npz (copied from the old repository; sha256 next to it): ARC-1 grids <= 10, transductive:
the train pairs of all 255 tasks are trained, their 278 test pairs evaluated. Canvas C = N + 1 = 11 (EOS row/column),
a lane is [input canvas, output canvas], T = 2 C^2 = 242 tokens (arc.model's tokens: 0 unknown, 1 pad, 2..11 colours,
12 EOS).

Episodes (CoupledEpisodes): a step is `batch` lanes in groups of up to D + 1 train pairs of random tasks (the last group
cut to fit), every lane a demonstration for its group and a query for itself (leave-one-out). URM/HRM augmentation:
per group a version v in 0..num_aug (0 identity; else a dihedral transform and a permutation of colours 1..9, fixed by
(seed, task, v) through Philox) and per lane an independent random translation inside the canvas (input and output
together). Evaluation: every test pair, the task's first D train pairs as demonstrations, version 0, no translation.

One step = one segment from a fresh carry (loops 1: Theta starts from 0 every step). Loss: stablemax CE on the query
lanes' output canvas (grid + EOS; pad ignored), mean per lane, mean over the batch (arc.train.lm_loss / batch: every
training lane is a query). AdamATan2 (0.9, 0.95; `adam` = 'old' is lt/train.py's form, 'trm' the adam-atan2 0.0.3 form
of arc.train), linear warmup then constant lr, weight decay on every parameter (decay_all, configurable), grad-norm
clip, a step with a non-finite gradient norm is skipped (as the old harness), eta projected after every step, EMA of
the parameters for evaluation. Evaluation every eval_every steps with the EMA weights: exact = every labelled cell
right (EOS included), cell = accuracy inside the true grid, cell_eos = with the EOS cells.

OUT/train.jsonl, OUT/eval.jsonl, OUT/config.json; OUT/last.pt (model, EMA, optimizer, step) at every evaluation.
Rerunning with the same --out resumes from last.pt (the data stream is reseeded with seed + step, as the old harness);
a different config is refused.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import log_stablemax
from .model import EOS, PAD, UNKNOWN, Coupled, ModelConfig
from .train import AdamATan2, decay_groups, flat_grads, lm_loss

ROOT = Path(__file__).resolve().parents[1]
FREE = ('steps', 'torch', 'gpu', 'eval_every', 'log_every', 'data')   # may differ on resume (data: data_sha decides)
DATA = ROOT / 'data' / 'arc1_ctx10_trans.npz'
DEFAULTS = dict(
    data=None,                        # default data/arc1_ctx10_trans.npz
    # ModelConfig fields; the old Gamma runs: d 512, 8 heads, bands 4, frames 8, the rest arc.model's defaults (= lt.coupled)
    model=dict(d=512, heads=8, blocks=8, grad_blocks=4, bands=4, frames=8,
               fused_bands=True, trace_dtype='float32', frame_bf16=False),
    D=5,                              # demonstration lanes per group (cmap's max_demos)
    steps=15000, batch=64, eval_groups=16,
    lr=3e-4, warmup=2000, beta1=0.9, beta2=0.95, weight_decay=0.1, ema=0.999, clip=1.0,
    decay_all=True, decay_lam=False,  # old harness: decay on every parameter
    adam='old',                       # 'old': lt/train.py AdamATan2 (the old runs); 'trm': arc.train's adam-atan2 0.0.3
    num_aug=1000, eval_every=1000, log_every=100, seed=0,
    holdout=False,                    # each task with >= 3 train pairs keeps its last one out of training (evaluated
                                      # as 'held'; the same tasks' trained pairs as 'seen')
    compile=True, cuda_graphs=True,
)


def load_config(src=None, overrides=()):
    cfg = json.loads(json.dumps(DEFAULTS))
    if src:
        user = {k: v for k, v in json.loads(Path(src).read_text()).items() if not k.startswith('_')}
        if 'model' in user:
            cfg['model'].update(user.pop('model'))
        cfg.update(user)
    for item in overrides:
        k, v = item.split('=', 1)
        v = json.loads(v)
        if k.startswith('model.'):
            cfg['model'][k[6:]] = v
        else:
            cfg[k] = v
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'unknown config keys {sorted(unknown)}')
    if cfg['adam'] not in ('old', 'trm'):
        raise ValueError("adam: 'old' or 'trm'")
    cfg['data'] = str(cfg['data'] or DATA)
    return cfg


class OldAdamATan2(AdamATan2):
    """lt/train.py's AdamATan2 (the old harness): p *= 1 - lr wd; m, v EMAs; p -= lr atan2(m / bc1, sqrt(v / bc2)).
    Differs from the adam-atan2 0.0.3 form only while the bias corrections matter (first ~50 steps)."""
    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            b1, b2 = g['betas']
            for p in g['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st.update(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                st['step'] += 1
                t, m, v = st['step'], st['m'], st['v']
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                p.mul_(1 - g['lr'] * g['weight_decay'])
                p.add_(torch.atan2(m / (1 - b1 ** t), (v / (1 - b2 ** t)).sqrt()), alpha=-g['lr'])


# ---------------------------------------------------------------------------------------------------- data
class Episodes:
    """lt.coupled.CoupledEpisodes of the old repository, unchanged in its draws (same rng calls in the same order), so
    a seed gives the old harness's batches."""
    def __init__(self, path, *, seed=0, num_aug=1000, D=5, N=None, holdout=False):
        with np.load(path, allow_pickle=False) as z:
            self.cells, self.off, self.hw, self.task = z['pair_cells'], z['pair_off'], z['pair_hw'], z['pair_task']
            self.trainable = np.asarray(z['pair_trainable'] if 'pair_trainable' in z
                                        else z['task_split'][z['pair_task']] == 0, bool)
        self.N = int(N or self.hw.max())
        self.C, self.D, self.nv, self.seed = self.N + 1, D, num_aug + 1, seed
        self.n_tasks = int(self.task.max()) + 1
        self.train_pairs = {t: np.flatnonzero((self.task == t) & self.trainable) for t in range(self.n_tasks)}
        self.test_pairs = {t: np.flatnonzero((self.task == t) & ~self.trainable) for t in range(self.n_tasks)}
        self.held = {}                  # holdout: each task with >= 3 train pairs loses its last one from training
        if holdout:
            for t in range(self.n_tasks):
                if len(self.train_pairs[t]) >= 3:
                    self.held[t] = int(self.train_pairs[t][-1])
                    self.train_pairs[t] = self.train_pairs[t][:-1]
        self.tasks = np.array([t for t in range(self.n_tasks) if len(self.train_pairs[t]) >= 2])
        self._key0 = np.uint64(0x9E3779B97F4A7C15 ^ int(seed))

    @property
    def n_test(self):
        return sum(len(v) for v in self.test_pairs.values())

    def grids(self, p):
        hi, wi, ho, wo = (int(x) for x in self.hw[p])
        o = int(self.off[p])
        return (self.cells[o:o + hi * wi].reshape(hi, wi).astype(np.int64),
                self.cells[o + hi * wi:o + hi * wi + ho * wo].reshape(ho, wo).astype(np.int64))

    def version(self, t, v):
        if v == 0:
            return 0, np.arange(10)
        key = np.array([self._key0, np.uint64(int(t) * self.nv + int(v))], dtype=np.uint64)
        g = np.random.Generator(np.random.Philox(key=key))
        return int(g.integers(0, 8)), np.concatenate([[0], 1 + g.permutation(9)])

    def canvas(self, grid, r, c):
        """C x C tokens: colour + 2 at offset (r, c), EOS on the row and column just outside, pad elsewhere."""
        C = self.C
        out = np.full((C, C), PAD, np.int64)
        h, w = grid.shape
        out[r:r + h, c:c + w] = grid + 2
        if r + h < C:
            out[r + h, c:min(c + w + 1, C)] = EOS
        if c + w < C:
            out[r:min(r + h + 1, C), c + w] = EOS
        return out

    def lane(self, p, k, perm, rng):
        gi, go = self.grids(p)
        gi, go = (np.rot90(perm[g], k % 4) for g in (gi, go))
        if k >= 4:
            gi, go = gi.T, go.T
        r = c = 0
        if rng is not None:                                     # URM: one translation per example
            r = int(rng.integers(0, self.C - max(gi.shape[0], go.shape[0]) + 1))
            c = int(rng.integers(0, self.C - max(gi.shape[1], go.shape[1]) + 1))
        x = np.concatenate([self.canvas(gi, r, c), np.full((self.C, self.C), UNKNOWN, np.int64)]).reshape(-1)
        y = np.concatenate([np.full((self.C, self.C), PAD, np.int64), self.canvas(go, r, c)]).reshape(-1)
        return x, y

    def group(self, t, pairs, v, rng, demo, query):
        k, perm = self.version(t, v)
        return [(*self.lane(p, k, perm, rng), d, q) for p, d, q in zip(pairs, demo, query)]

    def collate(self, groups, pad_to=None):
        """Lanes of the groups; pad_to: filler lanes (unknown inputs, pad labels, no demo, no query, each its own
        group) up to a fixed lane count, so the compiled layer sees one shape."""
        lanes = [(*x, g) for g, grp in enumerate(groups) for x in grp]
        T = 2 * self.C * self.C
        for i in range(len(lanes), pad_to or 0):
            lanes.append((np.full(T, UNKNOWN, np.int64), np.full(T, PAD, np.int64), False, False, len(groups) + i))
        col = lambda i, dt=None: torch.as_tensor(np.stack([l[i] for l in lanes]) if dt is None
                                                 else np.array([l[i] for l in lanes], dt))
        return dict(inputs=col(0), labels=col(1), demo=col(2, bool), query=col(3, bool), group=col(4, np.int64))

    def train_batch(self, n_lanes, rng):
        groups, n = [], 0
        while n < n_lanes:
            t = int(rng.choice(self.tasks))
            pairs = rng.permutation(self.train_pairs[t])[:min(self.D + 1, n_lanes - n)]
            groups.append(self.group(t, pairs, int(rng.integers(self.nv)), rng, [True] * len(pairs),
                                     [True] * len(pairs)))
            n += len(pairs)
        return self.collate(groups)

    def eval_batches(self, n_groups, kind='test'):
        """Every test pair: the task's first D train pairs (demonstrations) + the test pair (query), version 0; every
        batch padded to n_groups (D + 1) lanes. With holdout, kind 'held': each held-out train pair (never trained on)
        as the query, the task's remaining train pairs as demonstrations; kind 'seen': the same tasks' first remaining
        train pair (trained on) as the query, the other remaining ones + the held-out pair as demonstrations (as many
        demonstrations as 'held')."""
        if kind == 'test':
            items = [(t, p, list(self.train_pairs[t][:self.D])) for t in range(self.n_tasks) for p in self.test_pairs[t]]
        elif kind == 'held':
            items = [(t, p, list(self.train_pairs[t][:self.D])) for t, p in self.held.items()]
        elif kind == 'seen':
            items = [(t, int(self.train_pairs[t][0]), (list(self.train_pairs[t][1:]) + [p])[:self.D])
                     for t, p in self.held.items()]
        else:
            raise ValueError(kind)
        for i in range(0, len(items), n_groups):
            groups = []
            for t, p, demos in items[i:i + n_groups]:
                groups.append(self.group(t, demos + [p], 0, None, [True] * len(demos) + [False],
                                         [False] * len(demos) + [True]))
            yield self.collate(groups, pad_to=n_groups * (self.D + 1))


def scores(logits, labels, query):
    """Query lanes: exact, cell (inside the true grid), cell_eos (with EOS), as counts."""
    y = labels[query, -logits.shape[1]:]
    pred = logits[query].argmax(-1)
    lab = y != PAD
    grid = lab & (y != EOS)
    ok = pred == y
    return dict(exact=int(((ok | ~lab).all(-1)).sum()), n=len(y), cell=int((ok & grid).sum()), n_cell=int(grid.sum()),
                cell_eos=int((ok & lab).sum()), n_eos=int(lab.sum()))


def query_loss(logits, labels, query):
    """The old harness's loss: per query lane the mean stablemax CE (FP64) over labelled cells, mean over query lanes."""
    y = labels[query, -logits.shape[1]:].long()
    lab = (y != PAD).double()
    cell = -torch.gather(log_stablemax(logits[query].double()), -1, y[..., None]).squeeze(-1)
    return ((cell * lab).sum(-1) / lab.sum(-1).clamp_min(1)).mean()


# ---------------------------------------------------------------------------------------------------- training
class GraphedStep:
    """The step's segment (fresh carry), loss and backward pass as one CUDA graph over a static batch; gradients land
    in the fixed .grad views of flat_grads. One eager warm-up on the capture stream first."""
    def __init__(self, model, batch, n):
        self.model, self.n = model, n
        self.batch = {k: v.clone() for k, v in batch.items()}
        self.graph = None

    def _run(self):
        logits, _ = self.model.segment(None, self.batch)
        loss, _ = lm_loss(logits, self.batch['labels'], self.model.O)
        (loss / self.n).backward()
        return loss.detach() / self.n, logits.detach()

    def __call__(self, batch, flat):
        for k, v in batch.items():
            self.batch[k].copy_(v)
        if self.graph is None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._run()
            torch.cuda.synchronize()
            flat.zero_()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=s):
                self.out = self._run()
            torch.cuda.synchronize()
        self.graph.replay()
        return self.out


def evaluate(model, data, device, n_groups, kind='test'):
    was = model.training
    model.eval()
    tot = dict(exact=0, n=0, cell=0, n_cell=0, cell_eos=0, n_eos=0)
    ce, nq = 0.0, 0
    with torch.no_grad():
        for b in data.eval_batches(n_groups, kind):
            b = {k: v.to(device) for k, v in b.items()}
            logits, _ = model.segment(None, b)
            for k, v in scores(logits, b['labels'], b['query']).items():
                tot[k] += v
            q = int(b['query'].sum())
            ce += float(query_loss(logits, b['labels'], b['query'])) * q
            nq += q
    model.train(was)
    return dict(tot, test_ce=ce / max(nq, 1))


class EMAWeights:
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None, help='optional JSON with DEFAULTS keys')
    ap.add_argument('--out', required=True)
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='JSON values; model.X for model fields')
    ap.add_argument('--device', default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    probe = Episodes(cfg['data'], seed=cfg['seed'], num_aug=cfg['num_aug'], D=cfg['D'], holdout=cfg['holdout'])
    mcfg = dict(dict(canvas=probe.C), **cfg['model'])
    if mcfg['canvas'] != probe.C:
        raise ValueError(f"model.canvas {mcfg['canvas']} != N + 1 = {probe.C} of the data")
    torch.manual_seed(cfg['seed'])
    model = Coupled(ModelConfig(**mcfg)).to(device)
    data = probe
    Opt = OldAdamATan2 if cfg['adam'] == 'old' else AdamATan2
    opt = Opt(decay_groups(model, cfg), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))
    ema = {n: p.detach().clone() for n, p in model.named_parameters()}
    n_par = sum(p.numel() for p in model.parameters())
    sha = __import__('hashlib').sha256(Path(cfg['data']).read_bytes()).hexdigest()
    meta = dict(cfg, model=mcfg, data_sha=sha, params=n_par, tasks=len(data.tasks), test_pairs=data.n_test,
                torch=torch.__version__, gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu')
    start = 1
    ck_path = out / 'last.pt'
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        old = {k: v for k, v in ck['meta'].items() if k not in FREE}
        for k, v in DEFAULTS.items():               # keys added after the checkpoint was written: their defaults
            if k not in FREE:
                old.setdefault(k, v)
        new = {k: v for k, v in meta.items() if k not in FREE}
        if old != new:
            raise ValueError(f'{ck_path} was written with another config: '
                             f'{ {k: (old.get(k), new.get(k)) for k in set(old) | set(new) if old.get(k) != new.get(k)} }')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        ema = {n: t.to(device) for n, t in ck['ema'].items()}
        start = ck['step'] + 1
        print(f'[SMALL] resumed from step {ck["step"]} (data stream reseeded)', flush=True)
        del ck
    (out / 'config.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f"[SMALL] params {n_par:,} model {mcfg} tasks {meta['tasks']} test pairs {meta['test_pairs']} "
          f"device {meta['gpu']}", flush=True)

    cuda = device.type == 'cuda'
    if cfg['compile'] and cuda:
        import torch._dynamo.config as dc
        import torch._inductor.config as ic
        ic.triton.persistent_reductions = False
        dc.recompile_limit = max(dc.recompile_limit, 16)
        model.layer_fn = torch.compile(model.layer, dynamic=False)
    flat = flat_grads(model)
    graphed = cfg['cuda_graphs'] and cfg['compile'] and cuda and flat is not None
    params = list(model.parameters())
    rng = np.random.default_rng(cfg['seed'] + start)
    log, elog = open(out / 'train.jsonl', 'a'), open(out / 'eval.jsonl', 'a')
    step_fn = None
    t0, window = time.time(), []
    for step in range(start, cfg['steps'] + 1):
        ts = time.time()
        lr = cfg['lr'] * min(1.0, step / cfg['warmup'])
        for g in opt.param_groups:
            g['lr'] = lr
        b = {k: v.to(device, non_blocking=True) for k, v in data.train_batch(cfg['batch'], rng).items()}
        if graphed:
            if step_fn is None:
                step_fn = GraphedStep(model, b, cfg['batch'])
            loss, logits = step_fn(b, flat)
        else:
            logits, _ = model.segment(None, b)
            loss = query_loss(logits, b['labels'], b['query'])
            opt.zero_grad(set_to_none=flat is None)
            if flat is not None:
                flat.zero_()
            loss.backward()
            loss, logits = loss.detach(), logits.detach()
        if flat is not None:
            gn = flat.norm()
            flat.mul_(torch.clamp(cfg['clip'] / (gn + 1e-6), max=1.0))   # torch clip_grad_norm_'s rule
        else:
            gn = torch.nn.utils.clip_grad_norm_(params, cfg['clip'])
        if not torch.isfinite(gn):
            print(f'[SMALL] step {step} nonfinite grad, skipped', flush=True)
            if flat is not None:
                flat.zero_()
            continue
        opt.step()
        if flat is not None:
            flat.zero_()
        model.block.project_eta_(model.cfg)
        with torch.no_grad():
            ev = [ema[n] for n, _ in model.named_parameters()]
            torch._foreach_mul_(ev, cfg['ema'])
            torch._foreach_add_(ev, [p.detach() for p in params], alpha=1 - cfg['ema'])
        if cuda:
            torch.cuda.synchronize(device)
        window.append(time.time() - ts)
        if step % cfg['log_every'] == 0:
            s = scores(logits, b['labels'], b['query'])
            lam = float(model.cfg.lam_floor + torch.nn.functional.softplus(model.lam_raw.detach()))
            rec = dict(step=step, loss=float(loss), exact=s['exact'] / s['n'], cell=s['cell'] / max(s['n_cell'], 1),
                       cell_eos=s['cell_eos'] / max(s['n_eos'], 1), gnorm=float(gn), lr=lr, sec=time.time() - t0,
                       s_per_step=float(np.median(window)), tau=float(model.tau), lam=lam,
                       mem_gb=torch.cuda.max_memory_allocated(device) / 2 ** 30 if cuda else 0.0)
            window = []
            log.write(json.dumps(rec) + '\n')
            log.flush()
            print(f"[SMALL] step {step} loss {rec['loss']:.4f} cell {rec['cell']:.4f} exact {rec['exact']:.3f} "
                  f"gn {rec['gnorm']:.2f} tau {rec['tau']:.3f} lam {lam:.2e} {rec['s_per_step'] * 1e3:.1f} ms/step "
                  f"{rec['mem_gb']:.1f}GiB", flush=True)
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            te = time.time()
            with EMAWeights(model, ema):
                e = evaluate(model, data, device, cfg['eval_groups'])
                extra = {}
                for kind in ('held', 'seen') if cfg['holdout'] else ():
                    h = evaluate(model, data, device, cfg['eval_groups'], kind)
                    extra.update({f'{kind}_exact': h['exact'], f'{kind}_n': h['n'], f'{kind}_ce': h['test_ce']})
            rec = dict(step=step, exact=e['exact'], n=e['n'], cell=e['cell'] / max(e['n_cell'], 1),
                       cell_eos=e['cell_eos'] / max(e['n_eos'], 1), test_ce=e['test_ce'], sec=time.time() - t0,
                       eval_sec=time.time() - te, **extra)
            elog.write(json.dumps(rec) + '\n')
            elog.flush()
            print(f"[EVAL] step {step} exact {e['exact']}/{e['n']} cell {rec['cell']:.4f} "
                  f"cell_eos {rec['cell_eos']:.4f} test_ce {rec['test_ce']:.4f}"
                  + ''.join(f" {k} {rec[f'{k}_exact']}/{rec[f'{k}_n']} ({rec[f'{k}_ce']:.3f})" for k in ('held', 'seen') if f'{k}_n' in rec)
                  + f" ({rec['eval_sec']:.0f}s)", flush=True)
            tmp = ck_path.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(), ema=ema, opt=opt.state_dict(), step=step, meta=meta), tmp)
            tmp.replace(ck_path)
    log.close()
    elog.close()


if __name__ == '__main__':
    main()

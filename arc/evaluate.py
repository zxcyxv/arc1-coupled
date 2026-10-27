"""Evaluation in TRM's protocol (evaluators/arc.py) with the coupled model's own tie-break.

Every evaluation task in each of `versions` (0 = original): its demonstrations and its test pairs run as one group for
`segments` segments (16 = TRM's halt_max_steps at evaluation). Each test prediction is cropped (TRM _crop), mapped back
to the original frame (TRM inverse_aug) and hashed; per test input the candidates are ranked by (votes, mean
demonstration exactness, mean demonstration log-probability), where a version's demonstration scores come from the
same forward pass (each demonstration lane is predicted by the Theta of the others; no test answer is read). pass@k as
TRM: a task scores the fraction of its test inputs whose answer is among the top k, averaged over the 400 tasks.

    torchrun --standalone --nproc_per_node=4 -m arc.evaluate --ckpt runs/arc1/step_N.pt --versions all
    torchrun --standalone --nproc_per_node=4 -m arc.evaluate --ckpt runs/arc1/milestones/step_N.pt   (EMA weights)
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .data import ARC_MAX, Store, collate, crop, eval_groups, grid_hash, inverse
from .model import PAD, UNKNOWN

PASS_KS = (1, 2, 5, 10, 100, 1000)


def log_stablemax(x, dim=-1):
    s = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    return torch.log(s / s.sum(dim=dim, keepdim=True))


def eval_plan(groups, lanes):
    """Whole groups packed into micro-batches of at most `lanes` lanes: [[group, ...], ...]."""
    plan, cur, n = [], [], 0
    for g in groups:
        size = g[2] + g[3]
        if size > lanes:
            raise ValueError(f'a group of {size} lanes does not fit eval micro-batches of {lanes}')
        if n + size > lanes:
            plan.append(cur)
            cur, n = [], 0
        cur.append(g)
        n += size
    if cur:
        plan.append(cur)
    return plan


def eval_micro_batch(store, groups, lanes):
    """(batch with exactly `lanes` lanes, meta [(lane, gid, t, v, kind, j)]); padding lanes are neither demonstrations
    nor queries and sit in their own groups."""
    O = ARC_MAX * ARC_MAX
    rows, meta = [], []
    for gid, (t, v, nd, nt) in enumerate(groups):
        for j in range(nd):
            meta.append((len(rows), gid, t, v, 'demo', j))
            rows.append((*store.lane(t, v, j), gid, True, False))
        for j in range(nt):
            meta.append((len(rows), gid, t, v, 'test', j))
            rows.append((*store.lane(t, v, j, test=True), gid, False, True))
    blank_x = np.concatenate([np.full(O, PAD), np.full(O, UNKNOWN)])
    blank_y = np.full(2 * O, PAD)
    rows += [(blank_x, blank_y, len(groups) + i, False, False) for i in range(lanes - len(rows))]
    return collate(rows), meta


@torch.no_grad()
def predict(model, batch, segments):
    carry = None
    for _ in range(segments):
        logits, carry = model.segment(carry, batch)
    return logits


@torch.no_grad()
def run(model, store, versions, lanes, segments, rank=0, ws=1, device=torch.device('cpu'), log=print):
    """Distributed evaluation. Returns (metrics, records, grids) on rank 0, None elsewhere. records:
    [(t, j, v, hash, demo_exact, demo_logp)], grids: {hash: grid}."""
    was = model.training
    model.eval()
    plan = eval_plan(eval_groups(store, versions), lanes)
    recs, grids = [], {}
    t0 = time.time()
    for b in range(rank, len(plan), ws):
        batch, meta = eval_micro_batch(store, plan[b], lanes)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits = predict(model, batch, segments).float()
        y = batch['labels'][:, model.O:].long()
        lab = y != PAD
        pred = logits.argmax(-1)
        exact = ((pred == y) | ~lab).all(-1).cpu().numpy()
        colour = lab & (y >= 2) & (y <= 11)                           # the true grid's cells (EOS excluded)
        right = pred == y
        cells = torch.stack([(right & colour).sum(-1), colour.sum(-1), (right & lab).sum(-1), lab.sum(-1)], -1)
        cells = cells.cpu().numpy()
        lp = torch.gather(log_stablemax(logits.double()), -1, y[..., None]).squeeze(-1)
        logp = (torch.where(lab, lp, 0).sum(-1) / lab.sum(-1).clamp_min(1)).cpu().numpy()
        pred = pred.cpu().numpy()
        score = {}
        for lane, gid, t, v, kind, j in meta:
            if kind == 'demo':
                s = score.setdefault(gid, [0.0, 0.0, 0])
                s[0] += float(exact[lane])
                s[1] += float(logp[lane])
                s[2] += 1
        tests = [m for m in meta if m[4] == 'test']
        cropped = crop(pred[[m[0] for m in tests]].reshape(-1, ARC_MAX, ARC_MAX))
        for (lane, gid, t, v, kind, j), g in zip(tests, cropped):
            tid, mapping = store.version(t, v)
            g = inverse(g, tid, mapping) if g.size else g
            h = grid_hash(np.ascontiguousarray(g))
            grids.setdefault(h, g)
            de, dl, n = score.get(gid, [0.0, 0.0, 0])
            recs.append((t, j, v, h, de / max(n, 1), dl / max(n, 1), *(int(c) for c in cells[lane]), float(logp[lane])))
        if rank == 0 and (b // ws) % 200 == 0:
            log(f'[EVAL] micro-batch {b + 1}/{len(plan)} ({time.time() - t0:.0f}s)')
    model.train(was)
    if ws > 1:
        everything = [None] * ws if rank == 0 else None
        dist.gather_object((recs, grids), everything, dst=0)
        if rank:
            return None
        recs = [r for part in everything for r in part[0]]
        grids = {h: g for part in everything for h, g in part[1].items()}
    return score_records(store, recs, versions), recs, grids


def score_records(store, recs, versions):
    """pass@k (TRM: per task the fraction of its test inputs whose answer is in the top k, mean over tasks), the
    original version's exactness, the mean exactness over all evaluated views, and cell accuracies (inside the true
    grid, and with its EOS cells) of the original version and of all views. recs: (t, j, v, hash, demo exact, demo
    log-probability[, right grid cells, grid cells, right labelled cells, labelled cells[, the test lane's mean log-probability]])."""
    by_input = {}
    view_ok, ident = [], {}
    answers = {}
    cell = {'identity': np.zeros(4), 'view': np.zeros(4)}
    nll = {'identity': [], 'view': []}                                 # test lanes' mean CE over their labelled cells
    for r in recs:
        t, j, v, h, de, dl = r[:6]
        if len(r) > 10:
            nll['view'].append(-r[10])
            if v == 0:
                nll['identity'].append(-r[10])
        if len(r) > 6:
            cell['view'] += r[6:10]
            if v == 0:
                cell['identity'] += r[6:10]
        if (t, j) not in answers:
            answers[(t, j)] = grid_hash(store.answer(t, j))
        ok = h == answers[(t, j)]
        view_ok.append(ok)
        if v == 0:
            ident[(t, j)] = ok
        c = by_input.setdefault((t, j), {}).setdefault(h, [0, 0.0, 0.0])
        c[0] += 1
        c[1] += de
        c[2] += dl
    tasks = {}
    for (t, j) in sorted(answers):
        ranked = sorted(by_input[(t, j)].items(), key=lambda kv: (kv[1][0], kv[1][1] / kv[1][0], kv[1][2] / kv[1][0]),
                        reverse=True)
        hit = [answers[(t, j)] in [h for h, _ in ranked[:k]] for k in PASS_KS]
        tasks.setdefault(t, []).append(hit)
    n_tasks = len(store.eval_tasks)
    out = {f'pass@{k}': sum(np.mean([h[i] for h in hits]) for hits in tasks.values()) / n_tasks
           for i, k in enumerate(PASS_KS)}
    out.update(identity_exact=sum(ident.values()), identity_n=len(ident),
               identity_exact_task=(sum(np.mean([ident.get((t, j), False) for j in range(int(store.z['task_test_n'][t]))])
                                        for t in store.eval_tasks) / n_tasks) if ident else float('nan'),
               view_exact=float(np.mean(view_ok)) if view_ok else float('nan'),
               versions=len(versions), inputs=len(answers))
    for k, v in nll.items():
        if v:
            out[f'{k}_test_ce'] = float(np.mean(v))
    for k, c in cell.items():
        out[f'{k}_cell'] = float(c[0] / c[1]) if c[1] else float('nan')
        out[f'{k}_cell_eos'] = float(c[2] / c[3]) if c[3] else float('nan')
    return out


def submission(store, recs, grids, k=2):
    """ARC Prize format: {task: [{'attempt_1': grid, 'attempt_2': grid}, ...]} from the same ranking as pass@k."""
    by_input = {}
    for r in recs:
        t, j, v, h, de, dl = r[:6]
        c = by_input.setdefault((t, j), {}).setdefault(h, [0, 0.0, 0.0])
        c[0] += 1
        c[1] += de
        c[2] += dl
    out = {}
    for t in store.eval_tasks:
        name = str(store.names[t])
        out[name] = []
        for j in range(int(store.z['task_test_n'][t])):
            ranked = sorted(by_input.get((t, j), {}).items(),
                            key=lambda kv: (kv[1][0], kv[1][1] / kv[1][0], kv[1][2] / kv[1][0]), reverse=True)
            att = [grids[h].tolist() for h, _ in ranked[:k]] or [[[0]]]
            att += [att[0]] * (k - len(att))
            out[name].append({f'attempt_{i + 1}': a for i, a in enumerate(att)})
    return out


def version_set(store, spec):
    """'all' | 'identity' | N: versions 0..N-1 of every task (TRM's versions after 0 are its random augmentations in
    acceptance order, so the first N - 1 are a fixed random sample; a task with fewer versions uses what it has)."""
    if spec == 'all':
        return list(range(int(store.n_versions.max())))
    if spec == 'identity':
        return [0]
    return list(range(int(spec)))


def load_weights(model, ck, weights='ema'):
    """A full checkpoint (raw model + ema) or a milestone (model already holds the EMA weights) into model."""
    if 'ema' not in ck:                                              # milestone
        if ck.get('weights') != 'ema' or weights != 'ema':
            raise ValueError("a milestone holds the EMA weights only (use --weights ema)")
        model.load_state_dict(ck['model'])
        return
    model.load_state_dict(ck['model'])
    if weights == 'ema':
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(ck['ema'][n].to(p.dtype))


def main():
    from .train import build_model, init_distributed, load_config
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--versions', default='all', help="'all', 'identity' or a number of versions (0 + fixed others)")
    ap.add_argument('--lanes', type=int, default=96)
    ap.add_argument('--segments', type=int, default=None, help='default: the run config loops (16)')
    ap.add_argument('--weights', default='ema', choices=('ema', 'raw'))
    ap.add_argument('--out', default=None, help='default: <ckpt dir>/eval_<step>_<versions>.json')
    args = ap.parse_args()
    rank, ws, device = init_distributed()
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = load_config(ck['config'])
    store = Store(cfg['data'])
    model = build_model(cfg, device)
    load_weights(model, ck, args.weights)
    if cfg['compile'] and device.type == 'cuda':                     # as in training: one compiled block for all 8
        import torch._inductor.config as ic
        ic.triton.persistent_reductions = False
        model.layer_fn = torch.compile(model.layer, dynamic=False)        # one block and its rule head
    versions = version_set(store, args.versions)
    segments = args.segments or cfg['loops']
    t0 = time.time()
    res = run(model, store, versions, args.lanes, segments, rank, ws, device)
    if rank == 0:
        metrics, recs, grids = res
        metrics.update(step=ck['step'], weights=args.weights, segments=segments, seconds=time.time() - t0)
        out = Path(args.out or Path(args.ckpt).parent / f"eval_{ck['step']}_{args.versions}.json")
        out.write_text(json.dumps(metrics, indent=1) + '\n')
        out.with_suffix('.submission.json').write_text(json.dumps(submission(store, recs, grids)))
        print('[EVAL] ' + ' '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k} {v}' for k, v in metrics.items()),
              flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

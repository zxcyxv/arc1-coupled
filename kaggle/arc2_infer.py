# ---------------------------------------------------------------------------------------------------- inference
# ARC-AGI-2 inference in arc.evaluate's protocol: every task runs as one group (its demonstrations + its test inputs)
# in `versions` versions (the original + dihedral/colour augmentations), `segments` segments of the recursion each.
# Test predictions are cropped, mapped back to the original frame and voted; ties are broken by how well each version
# predicts its own demonstrations (exactness, then mean log-probability). No test-time training, no puzzle embedding.
import glob
import hashlib
import json
import os
import queue
import threading
import time

import numpy as np

ARC_MAX = 30
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]
TO_MODEL = np.array([PAD, EOS] + list(range(2, 12)), dtype=np.int64)     # TRM token (0 pad, 1 EOS, 2..11) -> model


def dihedral(arr, tid):
    if tid == 1:
        return np.rot90(arr, k=1)
    if tid == 2:
        return np.rot90(arr, k=2)
    if tid == 3:
        return np.rot90(arr, k=3)
    if tid == 4:
        return np.fliplr(arr)
    if tid == 5:
        return np.flipud(arr)
    if tid == 6:
        return arr.T
    if tid == 7:
        return np.fliplr(np.rot90(arr, k=1))
    return arr


def augment(grid, tid, mapping):
    return np.ascontiguousarray(dihedral(mapping[grid], tid))


def inverse(grid, tid, mapping):
    return np.ascontiguousarray(np.argsort(mapping).astype(np.uint8)[dihedral(grid, DIHEDRAL_INVERSE[tid])])


def canvas(grid, pr, pc):
    """TRM canvas (0 pad, 1 EOS, colour + 2) of one grid at offset (pr, pc)."""
    h, w = grid.shape
    out = np.zeros((ARC_MAX, ARC_MAX), dtype=np.uint8)
    out[pr:pr + h, pc:pc + w] = grid + 2
    if pr + h < ARC_MAX:
        out[pr + h, pc:pc + w] = 1
    if pc + w < ARC_MAX:
        out[pr:pr + h, pc + w] = 1
    return out


def crop(tokens):
    """TRM _crop over [N, 30, 30] model tokens: the largest top-left rectangle of colour cells, as colours 0..9."""
    tokens = np.asarray(tokens).reshape(-1, ARC_MAX, ARC_MAX)
    colour = (tokens >= 2) & (tokens <= 11)
    first_bad = np.where(colour.all(-1), ARC_MAX, np.argmin(colour, -1))
    cols = np.minimum.accumulate(first_bad, axis=1)
    area = cols * np.arange(1, ARC_MAX + 1)[None]
    rows = area.argmax(1) + 1
    best = area.max(1)
    out = []
    for n in range(len(tokens)):
        r, c = (int(rows[n]), int(cols[n, rows[n] - 1])) if best[n] > 0 else (0, 0)
        out.append((tokens[n, :r, :c] - 2).astype(np.uint8))
    return out


def grid_hash(grid):
    return hashlib.sha256(bytes(grid.shape) + np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest()


def log_stablemax(x, dim=-1):
    s = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    return torch.log(s / s.sum(dim=dim, keepdim=True))


def _rng(*key):
    return np.random.RandomState(int(hashlib.sha256('|'.join(map(str, key)).encode()).hexdigest()[:8], 16))


def task_versions(name, task, n, seed):
    """The original + n - 1 distinct augmentations (dihedral id, colour map with 0 fixed), as TRM's builder draws them;
    duplicates (same augmented grids) are skipped."""
    grids = [np.array(e[k], np.uint8) for e in task['train'] for k in ('input', 'output')]
    grids += [np.array(e['input'], np.uint8) for e in task['test']]
    key = lambda tid, m: hashlib.sha256(b''.join(bytes(a.shape) + a.tobytes()
                                                 for a in (augment(g, tid, m) for g in grids))).hexdigest()
    ident = np.arange(10, dtype=np.uint8)
    out, seen = [(0, ident)], {key(0, ident)}
    rs = _rng(seed, 'versions', name)
    for _ in range(50 * n):
        if len(out) >= n:
            break
        tid = int(rs.randint(0, 8))
        m = np.concatenate([[0], rs.permutation(np.arange(1, 10))]).astype(np.uint8)
        k = key(tid, m)
        if k not in seen:
            seen.add(k)
            out.append((tid, m))
    return out


def group_lanes(name, task, v, tid, mapping, seed):
    """Lanes of one task version: the demonstrations (TRM's training layout: one at the origin, the others at a random
    offset) and the test inputs (at the origin, labels unknown). [(inputs, labels, demo, query)]."""
    O = ARC_MAX * ARC_MAX
    rs = _rng(seed, 'pads', name, v)
    keep = int(rs.randint(0, len(task['train'])))
    lanes = []
    for j, e in enumerate(task['train']):
        i, o = augment(np.array(e['input'], np.uint8), tid, mapping), augment(np.array(e['output'], np.uint8), tid, mapping)
        if j == keep:
            pr = pc = 0
        else:
            pr = int(rs.randint(0, ARC_MAX - max(i.shape[0], o.shape[0]) + 1))
            pc = int(rs.randint(0, ARC_MAX - max(i.shape[1], o.shape[1]) + 1))
        x = np.concatenate([TO_MODEL[canvas(i, pr, pc).ravel()], np.full(O, UNKNOWN)])
        y = np.concatenate([np.full(O, PAD), TO_MODEL[canvas(o, pr, pc).ravel()]])
        lanes.append((x, y, True, False))
    for e in task['test']:
        i = augment(np.array(e['input'], np.uint8), tid, mapping)
        x = np.concatenate([TO_MODEL[canvas(i, 0, 0).ravel()], np.full(O, UNKNOWN)])
        lanes.append((x, np.full(2 * O, PAD), False, True))
    return lanes


def micro_batches(groups, lanes):
    """Whole groups packed into batches of at most `lanes` lanes (a larger group gets a batch of its own)."""
    plan, cur, n = [], [], 0
    for g in groups:
        size = len(g['lanes'])
        if cur and n + size > lanes:
            plan.append(cur)
            cur, n = [], 0
        cur.append(g)
        n += size
    if cur:
        plan.append(cur)
    return plan


def collate(batch_groups, lanes):
    """Tensors of the groups' lanes (in order), padded to `lanes` with blank lanes that sit in their own groups."""
    O = ARC_MAX * ARC_MAX
    rows = [(x, y, gid, demo, query) for gid, g in enumerate(batch_groups) for x, y, demo, query in g['lanes']]
    blank_x = np.concatenate([np.full(O, PAD), np.full(O, UNKNOWN)])
    for i in range(max(0, lanes - len(rows))):
        rows.append((blank_x, np.full(2 * O, PAD), len(batch_groups) + i, False, False))
    t = lambda k, dt=None: torch.as_tensor(np.stack([r[k] for r in rows]) if dt is None else np.array([r[k] for r in rows], dt))
    return dict(inputs=t(0), labels=t(1), group=t(2, np.int64), demo=t(3, bool), query=t(4, bool))


@torch.no_grad()
def run_batch(model, batch_groups, lanes, segments, device):
    batch = collate(batch_groups, lanes)
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    carry = None
    for _ in range(segments):
        logits, carry = model.segment(carry, batch)
    logits = logits.float()
    y = batch['labels'][:, model.O:].long()
    lab = y != PAD
    pred = logits.argmax(-1)
    exact = ((pred == y) | ~lab).all(-1).cpu().numpy()
    lp = torch.gather(log_stablemax(logits.double()), -1, y[..., None]).squeeze(-1)
    logp = (torch.where(lab, lp, 0).sum(-1) / lab.sum(-1).clamp_min(1)).cpu().numpy()
    pred = pred.cpu().numpy()
    out, row = [], 0
    for g in batch_groups:
        n = len(g['lanes'])
        demos = [r for r in range(row, row + n) if g['lanes'][r - row][2]]
        tests = [r for r in range(row, row + n) if g['lanes'][r - row][3]]
        de = float(np.mean(exact[demos])) if demos else 0.0
        dl = float(np.mean(logp[demos])) if demos else 0.0
        for j, g2 in enumerate(crop(pred[tests].reshape(-1, ARC_MAX, ARC_MAX)) if tests else []):
            if g2.size:
                g2 = inverse(g2, g['tid'], g['map'])
            out.append((g['name'], j, g2, de, dl))
        row += n
    return out


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    mcfg = dict(ck['config']['model'], fused_bands=False, trace_dtype='float32', dtype='float32')
    torch.manual_seed(0)
    model = Coupled(ModelConfig(**mcfg))
    state = ck['model'] if 'ema' not in ck else {k: ck['ema'].get(k, v) for k, v in ck['model'].items()}
    model.load_state_dict(state)
    return model.to(device).eval(), ck


def find(path, pattern):
    if path:
        return path
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True))
    if not hits:
        raise FileNotFoundError(f'no {pattern} under /kaggle/input')
    return hits[0]


def write_submission(path, tasks, results):
    """ARC Prize format from the (task, test input, grid, demo exactness, demo log-prob) records so far; every task and
    test input of the file is listed ([[0]] where nothing was predicted yet)."""
    votes, grids = {}, {}
    for name, j, g, de, dl in results:
        if not g.size:
            continue
        h = grid_hash(g)
        grids[h] = g
        c = votes.setdefault((name, j), {}).setdefault(h, [0, 0.0, 0.0])
        c[0] += 1
        c[1] += de
        c[2] += dl
    sub = {}
    for name in tasks:
        sub[name] = []
        for j in range(len(tasks[name]['test'])):
            ranked = sorted(votes.get((name, j), {}).items(),
                            key=lambda kv: (kv[1][0], kv[1][1] / kv[1][0], kv[1][2] / kv[1][0]), reverse=True)
            att = [grids[h].tolist() for h, _ in ranked[:2]] or [[[0]]]
            att += [att[0]] * (2 - len(att))
            sub[name].append({'attempt_1': att[0], 'attempt_2': att[1]})
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path + '.tmp', 'w') as f:
        json.dump(sub, f)
    os.replace(path + '.tmp', path)
    return sub


def main(cfg):
    import traceback
    t0 = time.time()
    ckpt = cfg['ckpt'] or (sorted(glob.glob('/kaggle/input/**/step_72224*.pt', recursive=True))
                           or sorted(glob.glob('/kaggle/input/**/*.pt', recursive=True)) or [None])[0]
    if ckpt is None:
        raise FileNotFoundError('attach the dataset with the checkpoint (step_72224.pt)')
    challenges = find(cfg['challenges'], 'arc-agi_test_challenges.json')
    with open(challenges) as f:
        tasks = json.load(f)
    names = list(tasks)[:cfg['max_tasks']] if cfg['max_tasks'] else list(tasks)
    devices = [torch.device('cuda', i) for i in range(torch.cuda.device_count())] or [torch.device('cpu')]
    print(f'[INFER] torch {torch.__version__}; {len(names)} tasks from {challenges}; checkpoint {ckpt}; devices '
          f'{[torch.cuda.get_device_name(d) if d.type == "cuda" else "cpu" for d in devices]}', flush=True)
    write_submission(cfg['out'], tasks, [])          # a valid file from the start: [[0]] everywhere
    print(f'[INFER] fallback submission written to {cfg["out"]}', flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    groups = []
    for name in names:
        for v, (tid, m) in enumerate(task_versions(name, tasks[name], cfg['versions'], cfg['seed'])):
            groups.append(dict(name=name, v=v, tid=tid, map=m,
                               lanes=group_lanes(name, tasks[name], v, tid, m, cfg['seed'])))
    plan = micro_batches(groups, cfg['lanes'])
    print(f'[INFER] {len(groups)} task versions, {sum(len(g["lanes"]) for g in groups)} lanes, '
          f'{len(plan)} micro-batches of up to {cfg["lanes"]} lanes, {cfg["segments"]} segments each', flush=True)

    # Preflight on the first device: load the checkpoint and run one full micro-batch for one segment, so a broken
    # setup fails within a minute, and time it for an estimate of the whole run.
    if devices[0].type == 'cuda':
        torch.cuda.set_device(devices[0])
    model0, ck = load_model(ckpt, devices[0])
    print(f'[INFER] checkpoint step {ck.get("step")}, blocks {model0.cfg.blocks}, canvas {model0.cfg.canvas}, '
          f'rope_1d {model0.cfg.rope_1d}, frames {model0.cfg.frames}, bf16 {model0.amp(devices[0])}', flush=True)
    for k in range(2):                               # the first pass includes CUDA start-up; time the second
        ts = time.time()
        run_batch(model0, plan[0], cfg['lanes'], 1, devices[0])
        if devices[0].type == 'cuda':
            torch.cuda.synchronize(devices[0])
    sec = time.time() - ts
    est = sec * cfg['segments'] * len(plan) / len(devices)
    print(f'[INFER] preflight ok: one micro-batch, one segment in {sec:.1f}s -> about {est / 60:.0f} min for '
          f'{len(plan)} micro-batches x {cfg["segments"]} segments on {len(devices)} device(s)', flush=True)

    work, results, lock, errors = queue.Queue(), [], threading.Lock(), []
    for b in plan:
        work.put(b)
    counter = {'done': 0}
    t1 = time.time()

    def worker(device, model):
        try:
            if device.type == 'cuda':
                torch.cuda.set_device(device)        # this thread's current GPU (allocations, bf16 check, streams)
            if model is None:
                model, _ = load_model(ckpt, device)
            while True:
                try:
                    b = work.get_nowait()
                except queue.Empty:
                    return
                out = run_batch(model, b, cfg['lanes'], cfg['segments'], device)
                with lock:
                    results.extend(out)
                    counter['done'] += 1
                    done = counter['done']
                    if done == 1 or done % 5 == 0 or done == len(plan):
                        el = time.time() - t1
                        print(f'[INFER] {done}/{len(plan)} micro-batches, {el / 60:.1f} min, '
                              f'eta {el / done * (len(plan) - done) / 60:.1f} min', flush=True)
                    if done % 20 == 0:                   # keep the file current in case the run is cut short
                        write_submission(cfg['out'], tasks, results)
        except Exception:
            msg = traceback.format_exc()
            print(f'[INFER] ERROR on {device}:\n{msg}', flush=True)
            errors.append((str(device), msg))

    threads = [threading.Thread(target=worker, args=(d, model0 if i == 0 else None)) for i, d in enumerate(devices)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    sub = write_submission(cfg['out'], tasks, results)
    n_pred = sum(1 for r in results if r[2].size)
    print(f'[INFER] wrote {cfg["out"]}: {len(sub)} tasks, {n_pred} non-empty predictions '
          f'({(time.time() - t0) / 60:.1f} min)', flush=True)
    if errors and counter['done'] < len(plan):
        raise RuntimeError(f'{len(plan) - counter["done"]} micro-batches were not run; see the errors above')
    if cfg['solutions']:
        with open(cfg['solutions']) as f:
            sols = json.load(f)
        score = [np.mean([any(a == sols[n][j] for a in (s['attempt_1'], s['attempt_2']))
                          for j, s in enumerate(sub[n])]) for n in names]
        print(f'[INFER] pass@2 (task-weighted) {np.mean(score):.4f} over {len(names)} tasks', flush=True)
    return sub


main(CFG)

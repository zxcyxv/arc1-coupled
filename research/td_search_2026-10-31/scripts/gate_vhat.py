"""Step 2 of TD + search (gate). Frozen checkpoint + the step-1 heads; nothing is trained. Test puzzles, FP32, one block
per call, noise off except where stated. Labels only score.

Search at a segment boundary b (b = 4, 8, 12 blocks done; s_b is the carry there):
- hypotheses: the least confident blank cell of the prediction at s_b x its top-K digits, each as a hypothetical input
  clue (only that input token changes; rule-free);
- branch k: a copy of the whole carry at s_b (hidden state, traces, W, C) with that input, rolled out R blocks;
- value of branch k: the R-step return of the same heads, -(1-g) sum_{j<R} g^j C^(s^k_{b+j}) + g^R V(s^k_{b+R}), the
  quantity V(s_b) estimates;
- V^(s_b) = max over the branches.
delta_{b+1} = -(1-g) C^(s_{b-1}) + g X - V(s_{b-1}), with X = V(s_b) (handoff) or V^(s_b) (TD + search). C^ and V are
board means of the per-cell heads, as in LT.tf_input.

A. Branch choice (all puzzles, noise off): among puzzles whose true digit is in the top-K, how often the highest-value
   branch holds it. References on the same leaves: the leaf V alone, and the lowest-entropy leaf (label-free).
   Head sanity: corr over puzzles of C^(s_b) with the true cell CE at s_b, and of V(s_b) with -L(s_{b+H}).
B. corr(delta, true advantage) over noise draws, as in the earlier diagnosis. Puzzles not exact at s_b; D draws of the
   v noise (sigma = tf_noise, as in training) in block b only, everything else noise-free. True advantage of draw d:
   -(L_d - mean_d L), L_d = mean cell CE (stablemax, true labels) of the root continued H blocks from s_b^(d).
   Per-puzzle correlation over the D draws, median over puzzles, for delta with V, with V^, and with the max leaf V
   alone (no path costs; the literal 'leaves scored by V').
Usage: python gate_vhat.py HEADS_CKPT OUT_JSON [K=3] [R=8] [D=16] [H=8] [NPUZ=512]
"""
import dataclasses, json, sys, time
import torch
sys.path.insert(0, '/workspace/arc1-coupled')
from lt.model import LT, ModelConfig
from lt.data import load_data

DEV = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
CK, OUT = sys.argv[1], sys.argv[2]
args = [int(x) for x in sys.argv[3:]]
K, R, D, H, NPUZ = args + [3, 8, 16, 8, 512][len(args):]
BOUNDS = (4, 8, 12)
CH = 8192
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'])
_, _, X, Y, _, _ = load_data(cfg['data_npz'], cfg['test_size'])
n = X.shape[-1]
T = n * n
cfg.update(batch_size=CH, amp=False, act_ckpt=False, blocks_per_seg=1)
m = LT(ModelConfig.from_dict(cfg)).to(DEV)
m.load_state_dict(ck['model'])
m.eval()
m.config.loops = 10 ** 6
m.config.tf_noise_eval = False
assert m.config.three_factor and float(m.block.tf_beta.abs().max()) == 0.0
g = m.config.tf_gamma
tok0 = torch.tensor(X.reshape(len(X), -1) + 1, dtype=torch.int32, device=DEV)
lab = torch.tensor(Y.reshape(len(Y), -1) + 1, dtype=torch.long, device=DEV)
N = len(tok0)


def bt(t):
    return dict(inputs=t, labels=torch.full_like(t, 2), puzzle_identifiers=torch.zeros(len(t), dtype=torch.int32, device=DEV))


def heads(h):
    c, v = m.tf_head(h)
    return c.double().mean(-1), v.double().mean(-1)


def probs(lg):
    x = lg.double()
    s = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    return s / s.sum(-1, keepdim=True)


def cell_ce(lg, y):
    return -torch.log(torch.gather(probs(lg), -1, y.unsqueeze(-1)).squeeze(-1).clamp_min(1e-300)).mean(-1)


def entropy(P, blank):
    pd = P[..., 2:n + 2].clamp_min(1e-12)
    e = -(pd * pd.log()).sum(-1)
    return (e * blank).sum(-1) / blank.sum(-1)


def sel_carry(c, idx=None, rep=1, inputs=None):
    """Lanes idx of carry c (all if None), each repeated rep times; inputs replaces the lanes' input tokens."""
    def f(x):
        if not torch.is_tensor(x):
            return x
        x = x if idx is None else x[idx]
        return x.repeat_interleave(rep, 0) if rep > 1 else x
    new = {fl.name: f(getattr(c, fl.name)) for fl in dataclasses.fields(c) if fl.name != 'data'}
    data = {k: f(v) for k, v in c.data.items()}
    if inputs is not None:
        data['inputs'] = inputs
    return dataclasses.replace(c, data=data, **new)


@torch.no_grad()
def search(carry, lg, tok, y):
    """Branches from the carry at s_b (B lanes, prediction lg, inputs tok). Returns per lane [B, K]: R-step return,
    leaf V, leaf entropy, leaf exact; and whether each branch's digit is the true one, and the true digit in the top-K."""
    B = len(tok)
    ar = torch.arange(B, device=DEV)
    pd = probs(lg)[..., 2:n + 2]
    conf = pd.max(-1).values.masked_fill(tok != 1, 2.0)
    cell = conf.argmin(-1)
    top = pd[ar, cell].topk(K, -1).indices
    br = tok[:, None, :].repeat(1, K, 1).clone()
    br[ar[:, None], torch.arange(K, device=DEV)[None, :], cell[:, None].expand(-1, K)] = (top + 2).int()
    br = br.view(-1, T)
    cb, bb = sel_carry(carry, rep=K, inputs=br), bt(br)
    c, _ = heads(cb.hidden)
    ret = -(1 - g) * c                                   # j = 0: C^(s_b), the same for every branch
    for j in range(1, R + 1):
        cb, lgb = m(cb, bb)
        c, v = heads(cb.hidden)
        ret = ret + (g ** R * v if j == R else -(1 - g) * g ** j * c)
    blank = (tok == 1).repeat_interleave(K, 0)
    yk = y.repeat_interleave(K, 0)
    true = (y[ar, cell] - 2)
    return dict(ret=ret.view(B, K), leafv=v.view(B, K), ent=entropy(probs(lgb), blank).view(B, K),
                exact=(lgb.argmax(-1) == yk).all(-1).view(B, K), is_true=top == true[:, None],
                has=(top == true[:, None]).any(1))


def pick_rate(score, d, mode):
    k = score.argmax(1) if mode == 'max' else score.argmin(1)
    right = d['is_true'][torch.arange(len(k), device=DEV), k] & d['has']
    return float(right.sum() / d['has'].sum().clamp_min(1))


def corr(a, b):
    a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
    return float((a * b).sum() / ((a * a).sum() * (b * b).sum()).sqrt().clamp_min(1e-300))


def pcorr(a, b):
    """Per-row correlation of [P, D] tensors (nan where either row is constant)."""
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    den = (a.pow(2).sum(1) * b.pow(2).sum(1)).sqrt()
    return torch.where(den > 1e-300, (a * b).sum(1) / den.clamp_min(1e-300), torch.full_like(den, float('nan')))


t0 = time.time()
res = dict(ckpt=CK, frozen_from=ck.get('frozen_from'), K=K, R=R, D=D, H=H, gamma=g, sigma=m.config.tf_noise, bounds={})
print(f'[GATE] {CK} (heads step {ck.get("step")}, frozen {ck.get("frozen_from")}); K {K}, R {R}, D {D}, H {H}, '
      f'g {g}, sigma {m.config.tf_noise}; {N} test puzzles', flush=True)
with torch.no_grad():
    # root, noise off: carries at b - 1 and b, logits at b, and L at b + H
    b0 = bt(tok0)
    carry = m.initial_carry(b0)
    keep, last = {}, max(BOUNDS) + H
    for j in range(1, last + 1):
        carry, lg = m(carry, b0)
        if j in BOUNDS or j + 1 in BOUNDS:
            keep[j] = (carry, lg)
        if j - H in BOUNDS:
            keep[f'L{j - H}'] = cell_ce(lg, lab)
    for b in BOUNDS:
        cb, lgb = keep[b]
        cprev, vprev = cb.tf_prev.double().unbind(-1)                    # heads on s_{b-1} (input of block b)
        c_b, v_b = heads(cb.hidden)
        root_exact = (lgb.argmax(-1) == lab).all(-1)
        out = dict(root_exact=int(root_exact.sum()),
                   corr_chat_vs_ce=corr(c_b, cell_ce(lgb, lab)), corr_v_vs_negL=corr(v_b, -keep[f'L{b}']))
        # ---------------- A. branch choice on all puzzles
        d = search(cb, lgb, tok0, lab)
        out.update(true_in_topk=float(d['has'].double().mean()),
                   pick_return=pick_rate(d['ret'], d, 'max'), pick_leafv=pick_rate(d['leafv'], d, 'max'),
                   pick_entropy=pick_rate(d['ent'], d, 'min'),
                   vhat_minus_v=float((d['ret'].max(1).values - v_b).mean()))
        # ---------------- B. corr(delta, true advantage) over noise draws in block b
        cpm, _ = keep[b - 1]
        idx = torch.nonzero(~root_exact).squeeze(1)[:NPUZ]
        P = CH // (D * K)
        cv, ch, cl, adv_all = [], [], [], []
        for s in range(0, len(idx), P):
            ii = idx[s:s + P]
            tk = tok0[ii].repeat_interleave(D, 0)
            yy = lab[ii].repeat_interleave(D, 0)
            c = sel_carry(cpm, ii, rep=D)
            m.config.tf_noise_eval = True                                  # block b only: independent draws per lane
            c, lgd = m(c, bt(tk))
            m.config.tf_noise_eval = False
            pc, pv = c.tf_prev.double().unbind(-1)
            _, vd = heads(c.hidden)
            cr = c
            for _ in range(H):
                cr, lgr = m(cr, bt(tk))
            L = cell_ce(lgr, yy)
            sd = search(c, lgd, tk, yy)
            vhat, vleaf = sd['ret'].max(1).values, sd['leafv'].max(1).values
            dv = -(1 - g) * pc + g * vd - pv
            dh = -(1 - g) * pc + g * vhat - pv
            dl = -(1 - g) * pc + g * vleaf - pv
            cv.append(dv.view(-1, D)); ch.append(dh.view(-1, D)); cl.append(dl.view(-1, D))
            adv_all.append(-(L - L.view(-1, D).mean(1).repeat_interleave(D)).view(-1, D))
        dv, dh, dl, adv = torch.cat(cv), torch.cat(ch), torch.cat(cl), torch.cat(adv_all)
        rv, rh, rl = pcorr(dv, adv), pcorr(dh, adv), pcorr(dl, adv)
        out.update(B_puzzles=len(idx), corr_delta_V=float(rv.nanmedian()), corr_delta_Vhat=float(rh.nanmedian()),
                   corr_delta_leafV=float(rl.nanmedian()),
                   frac_pos_V=float((rv > 0).double().mean()), frac_pos_Vhat=float((rh > 0).double().mean()),
                   adv_std=float(adv.std(1).median()), delta_V_std=float(dv.std(1).median()),
                   delta_Vhat_std=float(dh.std(1).median()))
        res['bounds'][b] = out
        print(f'== boundary {b} blocks: root exact {out["root_exact"]}/{N}; heads: corr(C^, CE) {out["corr_chat_vs_ce"]:+.3f}, '
              f'corr(V, -L(b+{H})) {out["corr_v_vs_negL"]:+.3f}', flush=True)
        print(f'   A. true digit in top-{K} {out["true_in_topk"]:.3f}; picks the true digit: R-step return {out["pick_return"]:.3f} | '
              f'leaf V {out["pick_leafv"]:.3f} | entropy {out["pick_entropy"]:.3f}; mean V^ - V {out["vhat_minus_v"]:+.4f}', flush=True)
        print(f'   B. {len(idx)} puzzles not exact, {D} draws: median corr(delta, true adv) with V {out["corr_delta_V"]:+.3f} '
              f'(pos {out["frac_pos_V"]:.2f}) | with V^ {out["corr_delta_Vhat"]:+.3f} (pos {out["frac_pos_Vhat"]:.2f}) | '
              f'with max leaf V {out["corr_delta_leafV"]:+.3f}; '
              f'std over draws: adv {out["adv_std"]:.2e}, delta(V) {out["delta_V_std"]:.2e}, delta(V^) {out["delta_Vhat_std"]:.2e} '
              f'({time.time() - t0:.0f}s)', flush=True)
res['seconds'] = time.time() - t0
json.dump(res, open(OUT, 'w'), indent=1)
print(f'[GATE] done in {res["seconds"]:.0f}s -> {OUT}', flush=True)

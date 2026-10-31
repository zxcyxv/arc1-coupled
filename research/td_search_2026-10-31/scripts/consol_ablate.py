"""Mechanism check on the trained consolidation model: is the third factor (delta) what carries the search result?
A = sum_k f(delta)_k (W_k - mean W) with f = identity (as trained), shuffled across the K branches, negated, or A = 0.
Also A built from the TRUE outcome (label-derived delta, diagnostic only) for the ceiling."""
import sys, json, torch
sys.argv = ['x', 'consol', '/tmp/claude-0/-workspace-arc1-coupled/d75a75c0-c7a3-4695-9045-6f537b19fbb0/scratchpad/abl', '1']
src = open('/tmp/claude-0/-workspace-arc1-coupled/d75a75c0-c7a3-4695-9045-6f537b19fbb0/scratchpad/consol_train.py').read()
exec(src.split("log = open(")[0])
sd = torch.load('/workspace/arc1-coupled/runs/consol/consol/final_ema.pt', map_location='cpu', weights_only=False)['model']
m.load_state_dict(sd)
m.eval(); m.config.tf_noise_eval = False
tok = torch.tensor(te_x.reshape(-1, 36) + 1, dtype=torch.int32, device=DEV)
lab = torch.tensor(te_y.reshape(-1, 36) + 1, dtype=torch.long, device=DEV)
N = len(tok)
res = {k: 0 for k in ('trained delta', 'shuffled delta', 'negated delta', 'A = 0', 'label delta (diagnostic)')}
g = torch.Generator(device='cpu').manual_seed(0)
with torch.no_grad():
    for i in range(0, N, B):
        t, y = tok[i:i + B], lab[i:i + B]; n = len(t)
        bb = batch_of(t); c = m.initial_carry(bb)
        for s in range(4):
            c, lg = m(c, bb)
        P = stab_probs(lg)
        pd = P[..., 2:8]
        conf = pd.max(-1).values.masked_fill(t != 1, 2.0)
        cell = conf.argmin(-1)
        top = pd[torch.arange(n), cell].topk(K, -1).indices
        br = t[:, None, :].repeat(1, K, 1).clone()
        br[torch.arange(n)[:, None], torch.arange(K)[None, :], cell[:, None].expand(-1, K)] = (top + 2).int()
        bbb = batch_of(br.view(-1, 36)); cb = m.initial_carry(bbb)
        for s in range(R // 4):
            cb, lgb = m(cb, bbb)
        Pb = stab_probs(lgb).view(n, K, 36, 8)
        ent = torch.stack([entropy(Pb[:, k], t) for k in range(K)], 1)
        d = -(ent - ent.mean(1, keepdim=True)); d = d / d.abs().amax(1, keepdim=True).clamp_min(1e-6)
        ok = (Pb.argmax(-1) == y[:, None]).all(-1).float()
        dl = ok - ok.mean(1, keepdim=True); dl = dl / dl.abs().amax(1, keepdim=True).clamp_min(1e-6)
        perm = torch.stack([torch.randperm(K, generator=g) for _ in range(n)]).to(DEV)
        Wk = cb.stdp_w.float().view(n, K, *c.stdp_w.shape[1:]); dev = Wk - Wk.mean(1, keepdim=True)
        mk = lambda dd: (dd.view(n, K, 1, 1, 1) * dev).sum(1)
        variants = {'trained delta': mk(d), 'shuffled delta': mk(d.gather(1, perm)), 'negated delta': mk(-d),
                    'A = 0': torch.zeros_like(c.stdp_w.float()), 'label delta (diagnostic)': mk(dl)}
        for name, A in variants.items():
            APERS['A'] = A
            c2 = c
            for s in range(2):
                c2, lg2 = m(c2, bb)
            res[name] += int((lg2.argmax(-1) == y).all(-1).sum())
            APERS['A'] = None
print(json.dumps(res))

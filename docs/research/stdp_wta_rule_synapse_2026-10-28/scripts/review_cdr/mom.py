exec(open('/tmp/claude-0/-workspace-arc1-coupled/ae142cb7-3355-47e0-b126-2adf442a38bb/scratchpad/theory2/toy_kernel.py').read().split("for T in (16, 32, 1000):")[0])
for T in (16, 32):
    for mode, kw in (('window', dict(beta=0.9, omega=0.5)), ('window', dict(beta=0.85, omega=0.5)),
                     ('inertial', dict(beta=0.7, omega=0.5)), ('inertial', dict(beta=0.8, omega=0.5)), ('inertial', dict(beta=0.9, omega=0.5)),
                     ('fista', dict(omega=0.5)), ('fista', dict(omega=0.4)), ('fista', dict(omega=0.6))):
        W, C = infer(U, V, mask, T, mode, **kw); ok = score(W, C)
        print(f'T={T} {mode:8s} {kw}: all {ok.mean():.3f} g0 {ok[group==0].mean():.3f} g1 {ok[group==1].mean():.3f} g2 {ok[group==2].mean():.3f}', flush=True)

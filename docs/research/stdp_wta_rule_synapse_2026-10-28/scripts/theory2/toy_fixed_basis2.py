exec(open('toy_fixed_basis.py').read().split("for mode, T, kw in")[0])
for mode, T, kw in (('fista', 100, dict(theta_w=1e-2)), ('fista', 300, dict(theta_w=1e-3)), ('fista', 1000, dict(theta_w=1e-3)), ('fista', 16, dict(theta_w=1e-2)),
                    ('circuit', 16, dict(theta_w=1e-2)), ('circuit', 100, dict(theta_w=1e-2)), ('circuit', 300, dict(theta_w=1e-3)), ('circuit', 1000, dict(theta_w=1e-3)),
                    ('circuit', 8, dict(theta_w=1e-3)), ('circuit', 32, dict(theta_w=1e-3)), ('circuit', 16, dict(omega=0.7)), ('circuit', 16, dict(omega=0.9)),
                    ('kwta', 16, dict(k=2)), ('kwta', 150, dict(k=2)), ('kwta', 150, dict(k=4))):
    t0 = time.time()
    W, C = infer(U, V, mask, mode, T, **kw)
    s = LR.read(Uq, W, C, Q); pred = torch.cat([torch.full_like(s[:, :1], 0.5), s], 1).argmax(1)
    ok = (pred == target).all(1).numpy()
    print(f'{mode:8s} T={T:4d} {kw}: exact g0 {ok[group == 0].mean():.3f} g1 {ok[group == 1].mean():.3f} g2 {ok[group == 2].mean():.3f} | W entries>0.05 median {(W.abs() > 0.05).sum((1, 2)).float().median().item():.0f} | {time.time() - t0:.0f}s', flush=True)

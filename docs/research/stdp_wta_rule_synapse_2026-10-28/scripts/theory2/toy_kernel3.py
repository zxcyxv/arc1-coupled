exec(open('toy_kernel.py').read().split("for T in (16, 32, 1000):")[0])
best = {}
for th in (0.05, 0.07, 0.1, 0.14, 0.2):
    W, C = infer(U, V, mask, 16, 'fista', omega=0.0, theta_w=th); ok = score(W, C)
    print(f'T=16 FISTA lasso theta_w={th}: exact all {ok.mean():.3f}', flush=True)
for th in (0.1, 0.2):
    W, C = infer(U, V, mask, 400, 'fista', omega=0.0, theta_w=th); ok = score(W, C)
    print(f'T=400 FISTA lasso theta_w={th} (fixed point at large lambda): exact all {ok.mean():.3f}', flush=True)
for beta in (0.8, 0.9):
    for om in (0.4, 0.5, 0.6):
        for th in (1e-3, 5e-3):
            W, C = infer(U, V, mask, 16, 'window', beta=beta, omega=om, theta_w=th); ok = score(W, C)
            print(f'T=16 circuit beta={beta} omega={om} theta_w={th}: exact all {ok.mean():.3f}', flush=True)

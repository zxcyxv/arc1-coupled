exec(open('toy_kernel.py').read().split("for T in (16, 32, 1000):")[0])
for T in (16,):
    for th in (3e-3, 1e-2, 3e-2, 1e-1, 3e-1):
        W, C = infer(U, V, mask, T, 'fista', omega=0.0, theta_w=th); ok = score(W, C)
        print(f'T={T} FISTA lasso only theta_w={th}: exact all {ok.mean():.3f}', flush=True)
    for th in (1e-2, 3e-2):
        W, C = infer(U, V, mask, T, 'window', beta=0.85, theta_w=th); ok = score(W, C)
        print(f'T={T} circuit beta 0.85 theta_w={th}: exact all {ok.mean():.3f}', flush=True)

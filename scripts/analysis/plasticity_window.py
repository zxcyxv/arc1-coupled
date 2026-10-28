"""The learned two-sided timing window of the bands memory, per head, from a checkpoint's band poles and coefficients.

Over blocks (the recurrent depth), E_K,r(t) = sum_{lag>=0} gamma_r z_r^lag k~(t - lag) and D_V,r(t) =
sum_{lag>=1} gamma_r z_r^lag v(t - lag), so S(t) = sum_r Re(a_r v^T E_K,r + b_r D_V,r^T k~) weights the outer product
post (v) x pre (k~) by
    pre leads post by lag >= 0:  K+(lag) = sum_r gamma_r Re(a_r z_r^lag)
    post leads pre by lag >= 1:  K-(lag) = sum_r gamma_r Re(b_r z_r^lag)
relative to the fixed +1 of the same-block term sum_t v k~^T in the read. W <- (1 - eta) W + eta S integrates S.

    python scripts/analysis/plasticity_window.py runs/small/b8g4_rope1d/last.pt checkpoints/run_X/step_N.pt ...
"""
import math
import sys

import torch


def state(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    ema = ck.get('ema')
    return ema if isinstance(ema, dict) and 'block.band_pole' in ema else ck['model']


def window(sd, lags=9):
    nu, th = sd['block.band_pole'].double()
    mag = torch.exp(-torch.exp(nu))
    z = mag * torch.exp(1j * th)
    g = torch.sqrt(-torch.expm1(-2 * torch.exp(nu)))
    ar, ai, br, bi = sd['block.band_coef'].double()
    lag = torch.arange(lags, dtype=torch.float64)
    zl = z[..., None] ** lag
    kp = (g[..., None] * (ar + 1j * ai)[..., None] * zl).real.sum(1)
    km = (g[..., None] * (br + 1j * bi)[..., None] * zl).real.sum(1)
    return kp, km[:, 1:], mag, th, torch.sigmoid(sd['block.eta_raw'].double())


def main():
    for path in sys.argv[1:]:
        kp, km, mag, th, eta = window(state(path))
        print(f'== {path}: head 0 periods {[round(2 * math.pi / abs(t), 1) for t in th[0].tolist()]}, '
              f'|z| {[round(m, 3) for m in mag[0].tolist()]}, eta {[round(e, 3) for e in eta.tolist()]}')
        for h in range(kp.shape[0]):
            print(f'  head {h}: pre->post lag 0..{kp.shape[1] - 1} ' + ' '.join(f'{v:+.2f}' for v in kp[h].tolist())
                  + f' | post->pre lag 1..{km.shape[1]} ' + ' '.join(f'{v:+.2f}' for v in km[h].tolist()))


if __name__ == '__main__':
    main()

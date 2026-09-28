import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from scipy.optimize import linear_sum_assignment




def calc_loss_base(x, model, device, config):
    x0 = torch.randn_like(x)
    t = torch.rand(x0.shape[0], device=x0.device)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x
    ut = x - x0 
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.training.autocast_bf16):
        vt = model(xt, t)
    return F.mse_loss(vt.float(), ut)


def calc_loss_ot(x, model, device, config):
    x0 = torch.randn_like(x)

    # Optimal transport cpu
    dists = torch.cdist(x0.flatten(1).float(), x.flatten(1).float()) ** 2
    rows, cols = linear_sum_assignment(dists.cpu().numpy())
    new_x0 = torch.empty_like(x0)
    new_x0[cols] = x0[rows]

    t = torch.rand(new_x0.shape[0], device=device)
    xt = (1 - t[:, None, None, None]) * new_x0 + t[:, None, None, None] * x
    ut = x - new_x0 
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.training.autocast_bf16):
        vt = model(xt, t)
    return F.mse_loss(vt.float(), ut)


def calc_loss_tpc(x, model, device, config):
    x0 = torch.randn_like(x)
    t = torch.rand(x0.shape[0], device=x0.device)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x
    xt_neg = t[:, None, None, None] * x0 + (1 - t[:, None, None, None]) * x
    ut = x - x0 

    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.training.autocast_bf16):
        v = model(torch.cat((xt, xt_neg)), torch.cat((t, 1-t)))
        vt, vt_neg = torch.chunk(v, 2)
    l = F.mse_loss(vt.float(), ut) + config.tpc.lambda_ * F.mse_loss(vt_neg.float(), vt.float())
    return l


def calc_loss_fdm(x, model, device, config):
    x0 = torch.randn_like(x)
    t = torch.rand(x0.shape[0], device=x0.device)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x
    ut = x - x0

    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.training.autocast_bf16):
        vt = model(xt, t)
    l_cfm = F.mse_loss(vt.float(), ut)

    n = config.fdm.num_div_samples
    xs, ts, x0s, us = xt[:n], t[:n], x0[:n], ut[:n]
    eps = torch.randint_like(xs, 2) * 2 - 1  # Rademacher probe for Hutchinson's trace estimator
    with sdpa_kernel(SDPBackend.MATH):
        vs, jvp = torch.func.jvp(lambda z: model.forward(z, ts), (xs,), (eps,))
    div_v = (eps * jvp).flatten(1).sum(1)
    d = xs[0].numel()
    err = (1 - ts) * div_v + d - ((vs - us) * x0s).flatten(1).sum(1)
    l_div = (err / d).abs().mean()

    return l_cfm + config.fdm.lambda_ * l_div


def calc_loss_bezier(x, model, device, config):
    x0 = torch.randn_like(x)
    t = torch.rand(x0.shape[0], device=x0.device)

    t0 = torch.zeros(x0.shape[0], device=x0.device)
    t1 = torch.ones(x0.shape[0], device=x0.device)

    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.training.autocast_bf16):
        with torch.no_grad():
            v0 = model(x0, t0)
            v1 = model(x, t1)
            xt, ut = cubic_bezier(x0, x, v0, v1, t[:, None, None, None])

        vt = model(xt, t)
    return F.mse_loss(vt.float(), ut)


def cubic_bezier(x0, x1, v0, v1, t):
    P0 = x0
    P1 = x0 + (1.0 / 3.0) * v0
    P2 = x1 - (1.0 / 3.0) * v1
    P3 = x1

    term0 = ((1 - t)**3) * P0
    term1 = 3.0 * ((1 - t)**2) * t * P1
    term2 = 3.0 * (1 - t) * (t**2) * P2
    term3 = (t**3) * P3
    x_t = term0 + term1 + term2 + term3

    d_term0 = 3.0 * ((1 - t)**2) * (P1 - P0)
    d_term1 = 6.0 * (1 - t) * t * (P2 - P1)
    d_term2 = 3.0 * (t**2) * (P3 - P2)
    v_t = d_term0 + d_term1 + d_term2
    return x_t, v_t


def gaussian_mixture():
    ...


loss_method_dict = {
    "base": calc_loss_base,
    "ot": calc_loss_ot,
    "tpc": calc_loss_tpc,
    "fdm": calc_loss_fdm,
}

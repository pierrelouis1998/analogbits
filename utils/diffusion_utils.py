import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils.bit_encoding import bit2rgb


def gamma(t, ns=0.0002, ds=0.00025):
    """Noising coefficient gamma as defined in the paper"""
    return np.cos(((t + ns) / (1 + ds)) * np.pi / 2) ** 2


def noise(t, eps, x_bits):
    """Noising process from x0 to xt"""
    x_crpt = torch.sqrt(gamma(t)) * x_bits + torch.sqrt(1 - gamma(t)) * eps
    return x_crpt


def ddim_step(x_t, x_pred, t_now, t_next):
    raise NotImplementedError


def ddpm_step(x_t, x_pred, t_now, t_next):
    raise NotImplementedError


def generate(steps: int, net: nn.Module, td=0, self_cond: bool = True, step_method='ddim'):
    """Sample from bit diffusion model"""
    net.eval()
    shape = net.input_shape  # usally (h, w, bitschannels)
    x_t = torch.normal(mean=0, std=1, size=shape, requires_grad=False)
    x_pred = torch.zeros_like(x_t, requires_grad=False)

    trange = tqdm(range(steps), desc="Sampling from xT ...")

    for step in trange:
        t_now = 1 - step / steps
        t_next = max(1 - (step + 1 + td) / steps, 0)

        # Predict x0
        if not self_cond:
            x_pred = torch.zeros_like(x_t, requires_grad=False)
        x_pred = net(torch.cat([x_t, x_pred], dim=-1))

        # Estimate x at t_next
        if step_method == 'ddim':
            x_t = ddim_step(x_t, x_pred, t_now, t_next)
        elif step_method == 'ddpm':
            x_t = ddpm_step(x_t, x_pred, t_now, t_next)

    return bit2rgb(x_pred > 0)

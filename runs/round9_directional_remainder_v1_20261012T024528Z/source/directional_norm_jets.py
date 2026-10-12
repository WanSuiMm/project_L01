"""Real-arithmetic directional Taylor inequalities, evaluated in float64.

Jet(v,t,B) represents f(0)=v, f'(0)=t, sup_[0,1]||f''||<=B.
This module does not implement outward rounding and never inspects Q endpoints.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import torch


def norm(x):
    return float(torch.linalg.vector_norm(x))


def bound_number(x):
    return float(x) if math.isfinite(float(x)) and 0 <= x < 1e100 else math.inf


@dataclass
class Jet:
    value: torch.Tensor
    tangent: torch.Tensor
    second: float = 0.

    @classmethod
    def constant(cls, value):
        return cls(value.detach(), torch.zeros_like(value), 0.)

    def __add__(self, other):
        return Jet(self.value + other.value, self.tangent + other.tangent,
                   bound_number(self.second + other.second))


@lru_cache(maxsize=128)
def opnorm(module):
    # Parameters are fixed in this experiment; cached modules retain identity.
    return float(torch.linalg.matrix_norm(module.weight.detach(), ord=2))


def linear(x, module):
    value = module(x.value).detach()
    tangent = module.weight.detach() @ x.tangent
    return Jet(value, tangent, bound_number(opnorm(module) * x.second))


def layernorm(x, module):
    d = x.value.numel()
    c0, c1 = x.value - x.value.mean(), x.tangent - x.tangent.mean()
    gamma = module.weight.detach()
    s0 = math.sqrt(float(c0.square().mean()) + module.eps)
    dot = float(c0 @ c1)
    tangent = gamma * (c1 / s0 - c0 * dot / (d * s0 ** 3))
    value = module(x.value).detach()
    B = x.second
    if not math.isfinite(B) or B > 1e30:
        return Jet(value, tangent, math.inf)
    c1norm, c0norm = norm(c1), norm(c0)
    theta_min = min(1., max(0., -dot / c1norm ** 2)) if c1norm else 0.
    segment_min = norm(c0 + theta_min * c1)
    segment_max = max(c0norm, norm(c0 + c1))
    Cmin, Cmax = max(0., segment_min - B / 2), segment_max + B / 2
    smin = math.sqrt(Cmin ** 2 / d + module.eps)
    G = c1norm + B
    gamma_max = float(gamma.abs().max())
    h_generic = 6 * gamma_max / (math.sqrt(d) * smin ** 2) * G ** 2
    # Enclose c(theta).c'(theta), retaining the known affine segment direction.
    dot_upper = max(abs(dot), abs(dot + c1norm ** 2)) + segment_max * B + B * c1norm / 2 + B ** 2 / 2
    h_direction = gamma_max * (2 * G * dot_upper / (d * smin ** 3)
                    + Cmax * G ** 2 / (d * smin ** 3)
                    + 3 * Cmax * dot_upper ** 2 / (d ** 2 * smin ** 5))
    second = gamma_max / smin * B + min(h_generic, h_direction)
    return Jet(value, tangent, bound_number(second))


def gelu(x):
    phi = torch.exp(-x.value.square() / 2) / math.sqrt(2 * math.pi)
    Phi = .5 * (1 + torch.erf(x.value / math.sqrt(2)))
    value = x.value * Phi
    tangent = (Phi + x.value * phi) * x.tangent
    if not math.isfinite(x.second) or x.second > 1e30:
        return Jet(value, tangent, math.inf)
    # |g'|<=1.25, |g''|<=0.8; ||(t+e)^2||2 <= ||t^2||2+2||t||inf||e||2+||e||2^2.
    squared_direction = norm(x.tangent.square()) + 2 * float(x.tangent.abs().max()) * x.second + x.second ** 2
    second = 1.25 * x.second + .8 * squared_direction
    return Jet(value, tangent, bound_number(second))


def attention(q, keys, values, weights0, weights1, d_model, n_full, n_summary):
    """Weighted measure quotient, with constant key gauge and centered numerator."""
    k0, k1 = torch.stack([k.value for k in keys]), torch.stack([k.tangent for k in keys])
    v0, v1 = torch.stack([v.value for v in values]), torch.stack([v.tangent for v in values])
    Bk = torch.tensor([k.second for k in keys], dtype=torch.float64)
    Bv = torch.tensor([v.second for v in values], dtype=torch.float64)
    # A fixed common key translation cancels from normalized attention for all theta.
    center = (weights0[:, None] * k0).sum(0) / weights0.sum()
    kc = k0 - center
    scale = math.sqrt(d_model)
    s0 = kc @ q.value / scale
    s1 = (kc @ q.tangent + k1 @ q.value) / scale
    exp0 = torch.exp(s0 - s0.max())
    exp0 = exp0 / (weights0 * exp0).sum()
    g0 = weights0 * exp0
    g1 = weights1 * exp0 + g0 * s1
    a0 = (g0[:, None] * v0).sum(0)
    U1 = (g1[:, None] * (v0 - a0)).sum(0) + (g0[:, None] * v1).sum(0)
    if not math.isfinite(q.second) or not bool(torch.isfinite(Bk).all()) or not bool(torch.isfinite(Bv).all()):
        return Jet(a0, U1, math.inf), {"reason": "upstream_second_derivative_unbounded"}
    q0norm, q1norm = norm(q.value), norm(q.tangent)
    k0norm = torch.linalg.vector_norm(kc, dim=1)
    k1norm = torch.linalg.vector_norm(k1, dim=1)
    # The known q1.k1 product is kept signed before bounding derivative variation.
    cross = (k1 @ q.tangent).abs() + q1norm * Bk + k1norm * q.second + q.second * Bk
    Bs = (q.second * (k0norm + k1norm + Bk / 2)
          + Bk * (q0norm + q1norm + q.second / 2) + 2 * cross) / scale
    score_radius = s1.abs() + Bs / 2
    if not bool(torch.isfinite(Bs).all()) or float(score_radius.max()) > 300:
        return Jet(a0, U1, math.inf), {"reason": "score_enclosure_not_numerically_evaluable",
                                           "max_score_radius": float(score_radius.max())}
    e_upper, e_lower = exp0 * torch.exp(score_radius), exp0 * torch.exp(-score_radius)
    Be = e_upper * (Bs + (s1.abs() + Bs).square())
    Ge = torch.minimum(e_upper * (s1.abs() + Bs), (exp0 * s1).abs() + Be)
    weight_max = torch.maximum(weights0, weights0 + weights1)
    Bg = weight_max * Be + 2 * weights1.abs() * Ge
    Gg = torch.minimum(weights1.abs() * e_upper + weight_max * Ge, g1.abs() + Bg)
    g_upper = weight_max * e_upper
    Gv = torch.linalg.vector_norm(v1, dim=1) + Bv
    Vmax = torch.linalg.vector_norm(v0 - a0, dim=1) + torch.linalg.vector_norm(v1, dim=1) + Bv / 2
    BU = float((Bg * Vmax + 2 * Gg * Gv + g_upper * Bv).sum())
    BD, D1 = float(Bg.sum()), float(g1.sum())
    start_suffix = n_full + n_summary
    lower_full = float((weights0[:n_full] * e_lower[:n_full]).sum())
    lower_summary = float((weights1[n_full:start_suffix] * e_lower[n_full:start_suffix]).sum())
    lower_suffix = float((weights0[start_suffix:] * e_lower[start_suffix:]).sum())
    Dmin = min(lower_full, lower_summary) + lower_suffix
    GU, GD, Umax = norm(U1) + BU, abs(D1) + BD, norm(U1) + BU / 2
    if Dmin <= 0 or not all(math.isfinite(x) for x in (BU, BD, GU, GD)):
        return Jet(a0, U1, math.inf), {"reason": "denominator_or_numerator_enclosure_unbounded"}
    try:
        B = BU / Dmin + 2 * GU * GD / Dmin ** 2 + Umax * BD / Dmin ** 2 + 2 * Umax * GD ** 2 / Dmin ** 3
    except (OverflowError, ZeroDivisionError):
        B = math.inf
    return Jet(a0, U1, bound_number(B)), {"reason": "finite_quotient_bound", "denominator_lower": Dmin,
                "denominator_second_upper": BD, "centered_numerator_second_upper": BU,
                "max_score_radius": float(score_radius.max())}


def rollout_bounds(model, full, summary, cont_id, history, n):
    caches, prefix_weights0, prefix_weights1 = [], [], []
    for (kp, vp, wp), (kq, vq, wq) in zip(full, summary):
        keys = [Jet.constant(v) for v in torch.cat((kp, kq))]
        values = [Jet.constant(v) for v in torch.cat((vp, vq))]
        caches.append((keys, values))
        prefix_weights0.append(torch.cat((wp, torch.zeros_like(wq))))
        prefix_weights1.append(torch.cat((-wp, wq)))
    all_logits, hidden, stages = [], [], []

    def retain(name, x, step, layer, details=None):
        stages.append({"stage": name, "step": step, "layer": layer, "tangent_norm": norm(x.tangent),
                       "second_upper": x.second, **(details or {})})
        return x

    for step, token in enumerate([cont_id, *history]):
        x = Jet.constant(model.token_embedding(torch.tensor(token)) + model.position_embedding(torch.tensor(n + step)))
        current_hidden = []
        for layer_id, layer in enumerate(model.layers):
            normalized = retain("attention_LN", layernorm(x, layer.norm_attn), step, layer_id)
            q, k, v = [linear(normalized, module) for module in [layer.q_proj, layer.k_proj, layer.v_proj]]
            keys, values = caches[layer_id]
            keys, values = [*keys, k], [*values, v]
            caches[layer_id] = keys, values
            w0 = torch.cat((prefix_weights0[layer_id], torch.ones(step + 1, dtype=torch.float64)))
            w1 = torch.cat((prefix_weights1[layer_id], torch.zeros(step + 1, dtype=torch.float64)))
            a, detail = attention(q, keys, values, w0, w1, model.d_model, len(full[layer_id][0]), len(summary[layer_id][0]))
            retain("attention", a, step, layer_id, detail)
            y = retain("attention_residual", x + linear(a, layer.out_proj), step, layer_id)
            normalized_ff = retain("FF_LN", layernorm(y, layer.norm_ff), step, layer_id)
            nonlinear = retain("GELU", gelu(linear(normalized_ff, layer.ff_in)), step, layer_id)
            x = retain("FF_residual", y + linear(nonlinear, layer.ff_out), step, layer_id)
            current_hidden.append(x)
        normalized_final = retain("final_LN", layernorm(x, model.final_norm), step, model.n_layers)
        W = model.lm_head.weight.detach()
        Wc = W - W.mean(0, keepdim=True)
        logits = Jet(Wc @ normalized_final.value, Wc @ normalized_final.tangent,
                     bound_number(float(torch.linalg.matrix_norm(Wc, ord=2)) * normalized_final.second))
        retain("centered_logits", logits, step, model.n_layers)
        all_logits.append(logits)
        hidden.append(current_hidden)
    return all_logits, hidden, stages

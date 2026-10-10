"""Apply the existing Round 6 global constants to a trained decoder.

This deliberately conservative, deterministic norm audit is separate from the
observed-query error diagnostics. It does not fit a constant to measured KL.
All exponential bounds are stored in log space, including vacuous bounds.
"""
from __future__ import annotations

import math
import torch


def opnorm(linear) -> float:
    return float(torch.linalg.matrix_norm(linear.weight.detach().double(), ord=2).cpu())


def biasnorm(linear) -> float:
    if linear.bias is None:
        return 0.0
    return float(torch.linalg.vector_norm(linear.bias.detach().double()).cpu())


def ln_constants(layer) -> tuple[float, float]:
    gamma = float(layer.weight.detach().abs().max().cpu())
    beta = float(torch.linalg.vector_norm(layer.bias.detach().double()).cpu())
    dimension = layer.weight.numel()
    # The centered normalization Jacobian has norm at most 1/sqrt(eps).
    return gamma / math.sqrt(layer.eps), math.sqrt(dimension) * gamma + beta


def logsum(values: list[float]) -> float:
    if not values:
        return -math.inf
    maximum = max(values)
    return maximum + math.log(sum(math.exp(x - maximum) for x in values))


def audit_model(model) -> dict:
    layers = []
    scale = model.d_model ** 0.25
    for layer in model.layers:
        ln_lip, ln_radius = ln_constants(layer.norm_attn)
        q, k, v = (opnorm(x) for x in (layer.q_proj, layer.k_proj, layer.v_proj))
        q_radius = (q * ln_radius + biasnorm(layer.q_proj)) / scale
        k_radius = (k * ln_radius + biasnorm(layer.k_proj)) / scale
        r = max(q_radius, k_radius)
        V = v * ln_radius + biasnorm(layer.v_proj)
        ff_lip, _ = ln_constants(layer.norm_ff)
        # sup |GELU'| < 1.13; use 1.2 without a fitted factor.
        beta = 1.0 + opnorm(layer.ff_out) * 1.2 * opnorm(layer.ff_in) * ff_lip
        sigma = beta * opnorm(layer.out_proj)
        wq, wk, wv = q * ln_lip / scale, k * ln_lip / scale, v * ln_lip
        B = r * r
        a = beta + 2.0 * sigma * V * r * wq
        log_b = math.log(sigma) + 2.0 * B + math.log(wv + 2.0 * V * r * wk)
        log_gain = logsum([math.log(a), math.log(2.0) + log_b])
        # epsZ <= eps and epsN <= V*eps give d=2*sigma*exp(B)*V*eps.
        log_d_per_epsilon = math.log(2.0 * sigma * V) + B
        layers.append(dict(r=r, value_radius=V, B=B, beta=beta, sigma=sigma,
                           wq=wq, wk=wk, wv=wv, a=a,
                           log_b=log_b, log_a_plus_2b=log_gain,
                           log_d_per_epsilon=log_d_per_epsilon))
    downstream, terms = 0.0, []
    for layer in reversed(layers):
        terms.append(layer["log_d_per_epsilon"] + downstream)
        downstream += layer["log_a_plus_2b"]
    final_lip, final_radius = ln_constants(model.final_norm)
    head_norm = opnorm(model.lm_head)
    W = head_norm * final_lip
    logit_abs_bound = head_norm * final_radius
    log_C = 2 * math.log(W) - math.log(4.0) + 2 * logsum(terms)
    return {
        "status": "CONSERVATIVE_GLOBAL_NORM_AUDIT",
        "layers": layers, "output_head_including_final_norm_lipschitz": W,
        "sampling_temperature": 1.0,
        "logit_absolute_bound": logit_abs_bound,
        "universal_per_token_KL_cap_from_logit_bound": math.log(model.vocab_size) + 2 * logit_abs_bound,
        "universal_cap_reason": "P(token|history) >= exp(-2*M)/vocab_size when all logits lie in [-M,M], hence KL(Q_t||P_t) <= log(vocab_size)+2*M for any Q_t.",
        "log_C_model_for_common_uniform_epsilon": log_C,
        "formula": "KL <= C_model * epsilon^2 * n*T/(n+T)",
        "scope": "Global LayerNorm and spectral-norm sufficient constants; no fitted gain and no empirical-query certificate.",
        "common_epsilon_certificate": "For a positive mass-n summary inside the original K/V norm bounds, epsilon <= 2*exp(max_layer_B). This certificate is conservative and can be vacuous.",
        "log_coarse_uniform_epsilon": math.log(2.0) + max(x["B"] for x in layers),
        "assumptions": "Positive weights, sum n, retained or convex-mean prefix K/V, unit-weight self-generated suffix, full softmax, tau=1. Unweighted-subset control is outside this certificate.",
        "numerical_note": "Spectral norms are evaluated in float64, not interval arithmetic. This is a numerical audit of sufficient analytic constants, not a machine-checked proof.",
    }

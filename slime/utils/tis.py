from typing import Any, Dict, Optional, Tuple

import torch


def masked_sum(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is None:
        return x.sum(dim=dim)
    return (x * mask).sum(dim=dim)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    denom = mask.sum(dim=dim).clamp_min(1)
    return masked_sum(x, mask, dim=dim) / denom


def compute_tis_weights(
    *,
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    eos_mask: Optional[torch.Tensor],
    level: str = "token",
    mode: str = "truncate",
    upper_threshold: Optional[float] = None,
    lower_threshold: Optional[float] = None,
    veto_threshold: float = 1e-4,
    safety_bound: float = 20.0,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """
    Compute the truncated importance sampling (TIS) weights and metrics.

    Adapted from:

    https://fengyao.notion.site/off-policy-rl#279721e3f6c48092bbe2fcfe0e9c6b33
    https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda

    Args:
        old_log_prob: The log probs from the policy model in the training backend. [batch_size, seq_len]
        rollout_log_prob: The log probs from the policy model in the rollout backend. [batch_size, seq_len]
        eos_mask: The mask of valid tokens. [batch_size, seq_len]
        level: The aggregation level for the importance sampling weights.
            - "token": per-token importance sampling weights, biased low variance.
            - "sequence": product over tokens, unbiased but high variance.
            - "geometric": geometric mean over tokens, biased, medium variance.
        mode: how to handle the importance sampling weights exceeding the thresholds.
            - "truncate": cap the importance sampling weights at the upper threshold, i.e., truncated importance sampling.
            - "clip": zero the importance sampling weights outside the [lower, upper] range.
        upper_threshold: The upper threshold for the importance sampling weights.
        lower_threshold: The lower threshold for the importance sampling weights, only used in "clip" mode.
            If not provided, it will be set to 1.0 / upper_threshold.
        veto_threshold: If any token's importance sampling weight is less than this, zero the entire sequence weight.
        safety_bound: The safety bound for the log-space ratio to avoid numerical overflow.

    Returns:
        weights: The importance sampling weights. [batch_size, seq_len]
        metrics: The metrics for the importance sampling weights.
    """
    if upper_threshold is None:
        return None, {}
    if lower_threshold is None:
        lower_threshold = 1.0 / upper_threshold

    device = old_log_prob.device
    log_ratio = old_log_prob - rollout_log_prob

    log_upper_threshold = torch.log(torch.tensor(upper_threshold, device=device))
    log_lower_threshold = torch.log(torch.tensor(lower_threshold, device=device))

    if level == "token":
        #  Token-level IS: π_training(a|s) / π_rollout(a|s) per token
        # The truncation will be applied later.
        log_ratio_for_metrics = log_ratio  # [batch_size, seq_len]
        log_ratio_safe = torch.clamp(log_ratio, min=-safety_bound, max=safety_bound)
        weights = torch.exp(log_ratio_safe)
    elif level == "sequence":
        # Sequence-level IS: π_training(a|s) / π_rollout(a|s) across the entire sequence
        log_ratio_sum = masked_sum(log_ratio, eos_mask, dim=-1).unsqueeze(-1)
        log_ratio_for_metrics = log_ratio_sum  # [batch_size, 1]
        log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-safety_bound, max=safety_bound)
        weights = torch.exp(log_ratio_sum_safe).expand_as(old_log_prob)
    elif level == "geometric":
        log_ratio_mean = masked_mean(log_ratio, eos_mask, dim=-1).unsqueeze(-1)
        log_ratio_for_metrics = log_ratio_mean  # [batch_size, 1]
        log_ratio_mean_safe = torch.clamp(log_ratio_mean, min=-safety_bound, max=safety_bound)
        weights = torch.exp(log_ratio_mean_safe).expand_as(old_log_prob)
    else:
        raise ValueError(f"Invalid importance sampling level: {level}")

    log_veto_threshold = torch.log(torch.tensor(veto_threshold, device=device))
    # Veto sequences with any token's log ratio below the threshold.
    # log(π_training / π_rollout) < log(veto_threshold) ⟺ π_training / π_rollout < veto_threshold
    catastrophic_tokens = (log_ratio < log_veto_threshold) & eos_mask.bool()
    has_catastrophic = catastrophic_tokens.any(dim=-1, keepdim=True)  # [batch_size, 1]
    veto_mask = (~has_catastrophic).float()  # [batch_size, 1]

    metrics = compute_is_metrics(
        is_weights=weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        eos_mask=eos_mask,
        level=level,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        log_upper_threshold=log_upper_threshold,
        log_lower_threshold=log_lower_threshold,
        has_catastrophic=has_catastrophic,
        catastrophic_tokens=catastrophic_tokens,
        safety_bound=safety_bound,
    )

    if mode == "truncate":
        # only truncate the weights at the upper threshold
        weights = weights.clamp(max=upper_threshold)
    elif mode == "clip":
        # zero the weights outside the [lower, upper] range
        if level in ["sequence", "geometric"]:
            seq_weights = weights[:, 0] if weights.dim() > 1 else weights
            sequence_clipped = ((seq_weights < lower_threshold) | (seq_weights > upper_threshold)).float()
            metrics["tis_sequence_clipped_fraction"] = sequence_clipped.mean()
        else:
            clip_mask = (weights >= lower_threshold) & (weights <= upper_threshold)
            clip_mask = clip_mask.float()
            clipped_indicator = 1 - clip_mask
            metrics["tis_token_clipped_fraction"] = masked_mean(clipped_indicator, eos_mask)
            sequence_has_clipped = masked_sum(clipped_indicator, eos_mask, dim=-1) > 0
            metrics["tis_sequence_clipped_fraction"] = sequence_has_clipped.float().mean()
        weights = weights * clip_mask
    else:
        raise ValueError(f"Invalid tis mode: {mode}")

    weights = weights * veto_mask
    weights = weights * eos_mask
    weights = weights.detach()

    metrics.update(
        {
            "tis_threshold_upper": upper_threshold,
            "tis_threshold_lower": lower_threshold,
            "tis_level": level,
            "tis_mode": mode,
            "tis_veto_threshold": veto_threshold,
        }
    )

    return weights, metrics


def compute_is_metrics(
    is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    eos_mask: Optional[torch.Tensor],
    *,
    level: str,
    upper_threshold: float,
    lower_threshold: float,
    log_upper_threshold: torch.Tensor,
    log_lower_threshold: torch.Tensor,
    has_catastrophic: Optional[torch.Tensor],
    catastrophic_tokens: Optional[torch.Tensor],
    safety_bound: float,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    if eos_mask is None:
        eos_mask = torch.ones_like(is_weights, dtype=torch.bool)

    device = is_weights.device

    if has_catastrophic is not None:
        metrics["tis_veto_fraction"] = has_catastrophic.float().mean()
    if catastrophic_tokens is not None and eos_mask is not None:
        metrics["tis_catastrophic_token_fraction"] = masked_mean(catastrophic_tokens.float(), eos_mask)

    if level in ["sequence", "geometric"]:
        log_max = log_ratio_for_metrics.max()
        log_min = log_ratio_for_metrics.min()
        metrics["tis_max"] = torch.exp(torch.clamp(log_max, max=safety_bound))
        metrics["tis_min"] = torch.exp(log_min)
        metrics["tis_mean"] = masked_mean(is_weights, eos_mask)
        exceeds_upper = log_ratio_for_metrics > log_upper_threshold
        below_lower = log_ratio_for_metrics < log_lower_threshold
        if level == "sequence":
            metrics["tis_ratio_fraction_high"] = exceeds_upper.float().mean()
            metrics["tis_ratio_fraction_low"] = below_lower.float().mean()
        else:
            exceeds_upper_exp = exceeds_upper.expand_as(eos_mask)
            below_lower_exp = below_lower.expand_as(eos_mask)
            metrics["tis_ratio_fraction_high"] = masked_mean(exceeds_upper_exp.float(), eos_mask)
            metrics["tis_ratio_fraction_low"] = masked_mean(below_lower_exp.float(), eos_mask)
    else:
        metrics["tis_mean"] = masked_mean(is_weights, eos_mask)
        above = is_weights > upper_threshold
        below = is_weights < lower_threshold
        metrics["tis_ratio_fraction_high"] = masked_mean(above.float(), eos_mask)
        metrics["tis_ratio_fraction_low"] = masked_mean(below.float(), eos_mask)
        if eos_mask.any():
            mask_bool = eos_mask.bool()
            metrics["tis_max"] = is_weights.masked_fill(~mask_bool, float("-inf")).max()
            metrics["tis_min"] = is_weights.masked_fill(~mask_bool, float("inf")).min()
        else:
            metrics["tis_max"] = torch.tensor(0.0, device=device)
            metrics["tis_min"] = torch.tensor(0.0, device=device)

    if eos_mask.any():
        weights_for_std = is_weights.clamp(min=lower_threshold, max=upper_threshold)
        var = masked_mean(weights_for_std.square(), eos_mask) - metrics["tis_mean"].square()
        metrics["tis_std"] = torch.sqrt(torch.clamp(var, min=0.0))
        weights_for_ess = weights_for_std / (metrics["tis_mean"] + 1e-8)
        metrics["tis_eff_sample_size"] = 1.0 / masked_mean(weights_for_ess.square(), eos_mask)
    else:
        metrics["tis_std"] = torch.tensor(0.0, device=device)
        metrics["tis_eff_sample_size"] = torch.tensor(1.0, device=device)

    if is_weights.dim() > 1 and eos_mask.any():
        seq_mean = masked_mean(is_weights, eos_mask, dim=-1)
        metrics["tis_seq_mean"] = seq_mean.mean()
        metrics["tis_seq_std"] = (
            seq_mean.std() if seq_mean.numel() > 1 else torch.tensor(0.0, device=is_weights.device)
        )
        metrics["tis_seq_max"] = seq_mean.max()
        metrics["tis_seq_min"] = seq_mean.min()
        seq_dev = (seq_mean - 1.0).abs()
        metrics["tis_seq_max_deviation"] = seq_dev.max()
        metrics["tis_seq_fraction_high"] = (seq_mean > upper_threshold).float().mean()
        metrics["tis_seq_fraction_low"] = (seq_mean < 1.0 / upper_threshold).float().mean()

    if eos_mask.any():
        flat = is_weights[eos_mask.bool()]
        if flat.numel() > 0:
            metrics["tis_p25"] = torch.quantile(flat, 0.25)
            metrics["tis_p50"] = torch.quantile(flat, 0.50)
            metrics["tis_p75"] = torch.quantile(flat, 0.75)
            metrics["tis_p95"] = torch.quantile(flat, 0.95)
            metrics["tis_p99"] = torch.quantile(flat, 0.99)

    return metrics


def compute_kl_metrics(
    *,
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    eos_mask: Optional[torch.Tensor],
    response_lengths: Optional[list[int]] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    device = old_log_prob.device
    if eos_mask is None:
        eos_mask = torch.ones_like(old_log_prob, dtype=torch.bool, device=device)

    # Direct estimator for KL(pi_rollout || pi_old): E[log pi_rollout - log pi_old]
    metrics["rollout_kl"] = masked_mean(rollout_log_prob - old_log_prob, eos_mask)

    # K3 estimator: E[exp(log(pi_old/pi_rollout)) - log(pi_old/pi_rollout) - 1]
    log_ratio = old_log_prob - rollout_log_prob
    k3_matrix = torch.exp(log_ratio) - log_ratio - 1
    metrics["rollout_k3_kl"] = masked_mean(k3_matrix, eos_mask)

    # Sequence-level perplexity difference metrics
    if old_log_prob.dim() == 2:
        mean_log_prob_rollout_per_seq = masked_mean(rollout_log_prob, eos_mask, dim=-1)
        mean_log_prob_old_per_seq = masked_mean(old_log_prob, eos_mask, dim=-1)
    elif response_lengths is not None and len(response_lengths) > 0 and old_log_prob.dim() == 1:
        seq_rollout_means = []
        seq_old_means = []
        start = 0
        for length in response_lengths:
            end = start + int(length)
            mask_chunk = eos_mask[start:end] if eos_mask is not None else None
            seq_rollout_means.append(masked_mean(rollout_log_prob[start:end], mask_chunk))
            seq_old_means.append(masked_mean(old_log_prob[start:end], mask_chunk))
            start = end
        mean_log_prob_rollout_per_seq = torch.stack(seq_rollout_means)
        mean_log_prob_old_per_seq = torch.stack(seq_old_means)
    else:
        # Fallback to global means if sequence boundaries are unavailable
        mean_log_prob_rollout_per_seq = masked_mean(rollout_log_prob, eos_mask).unsqueeze(0)
        mean_log_prob_old_per_seq = masked_mean(old_log_prob, eos_mask).unsqueeze(0)

    diff = mean_log_prob_rollout_per_seq - mean_log_prob_old_per_seq
    metrics["log_ppl_diff"] = diff.mean()
    metrics["log_ppl_abs_diff"] = diff.abs().mean()

    return metrics

from typing import Any, Dict, Optional, Tuple

import torch


def masked_sum(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes the sum of the tensor x, masked by the mask.

    x = [[1, 2, 3], [4, 5, 6]]
    mask = [[1, 1, 1], [1, 1, 0]]
    masked_sum(x, mask, dim=-1) = [6, 9]
    """
    valid_tokens = mask.sum(dim=dim)
    assert valid_tokens.min() > 0, "any sequence must have at least one valid token"
    assert x.shape == mask.shape, "x and mask must have the same shape"
    return (x * mask).sum(dim=dim)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes the mean of the tensor x, masked by the mask.

    x = [[1, 2, 3], [4, 5, 6]]
    mask = [[1, 1, 1], [1, 1, 0]]
    masked_mean(x, mask, dim=-1) = [2, 4.5]
    """
    valid_tokens = mask.sum(dim=dim)
    assert valid_tokens.min() > 0, "any sequence must have at least one valid token"
    return masked_sum(x, mask, dim=dim) / valid_tokens


def per_seq_masked_mean(
    x: torch.Tensor,
    mask: torch.Tensor,
    response_lengths: Optional[list[int]] = None,
) -> torch.Tensor:
    """
    计算按样本的 masked mean 后再求和，返回一个可加性的标量（适配 DP 汇总）。
    支持二维 [B, T] 与拍平后一维、并提供 response_lengths 的两种输入形态。
    """
    if response_lengths is not None and len(response_lengths) > 0:
        sequence_log_ratios = torch.split(x, [int(l) for l in response_lengths], dim=0)
        sequence_loss_masks = torch.split(mask, [int(l) for l in response_lengths], dim=0)
        seq_means = [
            masked_mean(sequence_log_ratio, sequence_loss_mask)
            for sequence_log_ratio, sequence_loss_mask in zip(sequence_log_ratios, sequence_loss_masks)
        ]
        return torch.stack(seq_means).sum()
    # fallback:视为单一样本
    return masked_mean(x, mask).unsqueeze(0).sum()


def compute_tis_weights(
    *,
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    loss_mask: torch.Tensor,
    level: str = "token",
    mode: str = "truncate",
    upper_threshold: Optional[float] = None,
    lower_threshold: Optional[float] = None,
    veto_threshold: float = 1e-4,
    safety_bound: float = 20.0,
    response_lengths: Optional[list[int]] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """
    Compute the truncated importance sampling (TIS) weights and metrics.

    Adapted from:

    https://fengyao.notion.site/off-policy-rl#279721e3f6c48092bbe2fcfe0e9c6b33
    https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda

    Args:
        old_log_prob: Flattened log probs from training backend. Shape: [sum(response_lengths)]
        rollout_log_prob: Flattened log probs from rollout backend. Shape: [sum(response_lengths)]
        loss_mask: Flattened mask aligned with flattened tensors. Shape: [sum(response_lengths)]
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
    assert (
        loss_mask.shape == old_log_prob.shape and loss_mask.shape == rollout_log_prob.shape
    ), "loss_mask, old_log_prob, and rollout_log_prob must have the same shape"
    assert response_lengths is not None and len(response_lengths) > 0, "response_lengths must be provided"

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
        log_ratio_for_metrics = log_ratio  # [sum(response_lengths)]
        log_ratio_safe = torch.clamp(log_ratio, min=-safety_bound, max=safety_bound)
        weights = torch.exp(log_ratio_safe)
    elif level in ["sequence", "geometric"]:
        # Sequence-level/geometric: compute per-sequence aggregate in log-space, then expand to tokens
        sequence_log_ratios = torch.split(log_ratio, [int(l) for l in response_lengths], dim=0)
        sequence_loss_masks = torch.split(loss_mask, [int(l) for l in response_lengths], dim=0)
        per_seq_vals = []
        for sequence_log_ratio, sequence_loss_mask in zip(sequence_log_ratios, sequence_loss_masks):
            if level == "sequence":
                val = (sequence_log_ratio * sequence_loss_mask).sum()
            else:  # geometric
                val = masked_mean(sequence_log_ratio, sequence_loss_mask)
            per_seq_vals.append(torch.clamp(val, min=-safety_bound, max=safety_bound))
        per_seq_vals = torch.stack(per_seq_vals)  # [num_sequences]
        per_seq_weights = torch.exp(per_seq_vals)
        # Expand to per-token weights per sequence
        expanded = []
        for w, sequence_log_ratio in zip(per_seq_weights, sequence_log_ratios):
            expanded.append(torch.ones_like(sequence_log_ratio) * w)
        weights = torch.cat(expanded, dim=0)
        # For metrics that need the aggregated log-ratio, keep per-seq values
        log_ratio_for_metrics = per_seq_vals
    else:
        raise ValueError(f"Invalid importance sampling level: {level}")

    log_veto_threshold = torch.log(torch.tensor(veto_threshold, device=device))
    # Veto sequences with any token's log ratio below the threshold.
    # log(π_training / π_rollout) < log(veto_threshold) ⟺ π_training / π_rollout < veto_threshold
    catastrophic_tokens = (log_ratio < log_veto_threshold) & loss_mask.bool()
    # Build per-sequence veto and expand to tokens
    cat_chunks = torch.split(catastrophic_tokens, [int(l) for l in response_lengths], dim=0)
    has_catastrophic_per_seq = torch.tensor([chunk.any() for chunk in cat_chunks], device=device)
    veto_mask = torch.cat(
        [
            torch.zeros_like(chunk, dtype=torch.float32) if has_cat else torch.ones_like(chunk, dtype=torch.float32)
            for has_cat, chunk in zip(has_catastrophic_per_seq, cat_chunks)
        ],
        dim=0,
    )

    metrics = compute_tis_metrics(
        tis_weights=weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        loss_mask=loss_mask,
        level=level,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        log_upper_threshold=log_upper_threshold,
        log_lower_threshold=log_lower_threshold,
        has_catastrophic=has_catastrophic_per_seq,
        catastrophic_tokens=catastrophic_tokens,
        safety_bound=safety_bound,
        response_lengths=response_lengths,
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
            metrics["tis_token_clipped_fraction"] = masked_mean(clipped_indicator, loss_mask)
            sequence_has_clipped = masked_sum(clipped_indicator, loss_mask, dim=-1) > 0
            metrics["tis_sequence_clipped_fraction"] = sequence_has_clipped.float().mean()
        weights = weights * clip_mask
    else:
        raise ValueError(f"Invalid tis mode: {mode}")

    weights = weights * veto_mask
    weights = weights * loss_mask
    weights = weights.detach()

    metrics["tis_threshold_upper"] = upper_threshold
    metrics["tis_threshold_lower"] = lower_threshold
    metrics["tis_level"] = level
    metrics["tis_mode"] = mode
    metrics["tis_veto_threshold"] = veto_threshold
    return weights, metrics


def compute_tis_metrics(
    *,
    tis_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    loss_mask: torch.Tensor,
    level: str,
    upper_threshold: float,
    lower_threshold: float,
    log_upper_threshold: torch.Tensor,
    log_lower_threshold: torch.Tensor,
    has_catastrophic: torch.Tensor,
    catastrophic_tokens: torch.Tensor,
    safety_bound: float,
    response_lengths: Optional[list[int]] = None,
) -> Dict[str, Any]:
    """
    Computes metrics that reflect the TRUE distribution (before clamping)
    for the truncated importance sampling (TIS) weights.
    """
    metrics: Dict[str, Any] = {}

    assert loss_mask.shape == tis_weights.shape, "loss_mask and tis_weights must have the same shape"

    # Counts/fractions reported as sum over sequences; external reducer divides by num_samples
    metrics["tis_veto_fraction"] = has_catastrophic.float().sum()
    metrics["tis_catastrophic_token_fraction"] = per_seq_masked_mean(
        catastrophic_tokens.float(), loss_mask, response_lengths=response_lengths
    )
    metrics["tis_level"] = level
    assert upper_threshold == 2.0
    # Make numeric constants DP-safe by scaling with number of sequences in this batch
    if tis_weights.dim() == 2:
        num_sequences = tis_weights.size(0)
    elif response_lengths is not None and len(response_lengths) > 0:
        num_sequences = len(response_lengths)
    else:
        num_sequences = 1
    metrics["tis_upper_threshold"] = torch.tensor(2.0 * num_sequences, device=tis_weights.device)
    metrics["tis_lower_threshold"] = torch.tensor(lower_threshold * num_sequences, device=tis_weights.device)
    metrics["tis_log_upper_threshold"] = log_upper_threshold * num_sequences
    metrics["tis_log_lower_threshold"] = log_lower_threshold * num_sequences
    metrics["tis_safety_bound"] = torch.tensor(safety_bound * num_sequences, device=tis_weights.device)

    if level in ["sequence", "geometric"]:
        # log_ratio_for_metrics is per-seq aggregated log-ratio: compare per-seq
        exceeds_upper = (log_ratio_for_metrics > log_upper_threshold).float().sum()
        below_lower = (log_ratio_for_metrics < log_lower_threshold).float().sum()
        metrics["tis_ratio_fraction_exceeds_upper"] = exceeds_upper
        metrics["tis_ratio_fraction_below_lower"] = below_lower
        metrics["tis_mean"] = per_seq_masked_mean(tis_weights, loss_mask, response_lengths=response_lengths)
    else:
        metrics["tis_mean"] = per_seq_masked_mean(tis_weights, loss_mask, response_lengths=response_lengths)
        exceeds_upper = (tis_weights > upper_threshold).float()
        below_lower = (tis_weights < lower_threshold).float()
        metrics["tis_ratio_fraction_exceeds_upper"] = per_seq_masked_mean(
            exceeds_upper, loss_mask, response_lengths=response_lengths
        )
        metrics["tis_ratio_fraction_below_lower"] = per_seq_masked_mean(
            below_lower, loss_mask, response_lengths=response_lengths
        )

    # Per-sequence std and ESS, reported as sum across sequences
    weights_for_std = tis_weights.clamp(min=lower_threshold, max=upper_threshold)
    sequence_log_ratios = torch.split(tis_weights, [int(l) for l in response_lengths], dim=0)
    sequence_loss_masks = torch.split(loss_mask, [int(l) for l in response_lengths], dim=0)
    per_seq_mean = torch.stack(
        [
            masked_mean(sequence_log_ratio, sequence_loss_mask)
            for sequence_log_ratio, sequence_loss_mask in zip(sequence_log_ratios, sequence_loss_masks)
        ]
    )
    per_seq_var = (
        torch.stack(
            [
                masked_mean(
                    sequence_log_ratio.clamp(min=lower_threshold, max=upper_threshold).square(), sequence_loss_mask
                )
                for sequence_log_ratio, sequence_loss_mask in zip(sequence_log_ratios, sequence_loss_masks)
            ]
        )
        - per_seq_mean.square()
    )
    per_seq_std = torch.sqrt(torch.clamp(per_seq_var, min=0.0))
    metrics["tis_std"] = per_seq_std.sum()
    # ESS per sequence using normalized weights
    weights_for_ess_list = [
        sequence_log_ratio / (pm + 1e-8) for sequence_log_ratio, pm in zip(sequence_log_ratios, per_seq_mean)
    ]
    per_seq_ess = torch.stack(
        [
            1.0 / masked_mean(sequence_log_ratio.square(), sequence_loss_mask)
            for sequence_log_ratio, sequence_loss_mask in zip(weights_for_ess_list, sequence_loss_masks)
        ]
    )
    metrics["tis_eff_sample_size"] = per_seq_ess.sum()
    seq_mean = per_seq_mean

    # Sequence-level summaries (sum-style for DP-safe reduction)
    metrics["tis_seq_mean"] = seq_mean.sum()
    metrics["tis_seq_fraction_exceeds_upper"] = (seq_mean > upper_threshold).float().sum()
    metrics["tis_seq_fraction_below_lower"] = (seq_mean < lower_threshold).float().sum()

    return metrics


def compute_kl_metrics(
    *,
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    response_lengths: Optional[list[int]] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    device = old_log_prob.device
    if loss_mask is None:
        loss_mask = torch.ones_like(old_log_prob, dtype=torch.bool, device=device)

    # Direct estimator for KL(pi_rollout || pi_old): per-seq mean then sum (1D inputs only)
    assert response_lengths is not None and loss_mask is not None
    sequence_log_ratios = torch.split(rollout_log_prob - old_log_prob, [int(l) for l in response_lengths], dim=0)
    sequence_loss_masks = torch.split(loss_mask, [int(l) for l in response_lengths], dim=0)
    per_seq = [
        masked_mean(sequence_log_ratio, sequence_loss_mask)
        for sequence_log_ratio, sequence_loss_mask in zip(sequence_log_ratios, sequence_loss_masks)
    ]
    metrics["rollout_kl"] = torch.stack(per_seq).sum()

    # K3 estimator: E[exp(log(pi_old/pi_rollout)) - log(pi_old/pi_rollout) - 1]
    log_ratio = old_log_prob - rollout_log_prob
    k3_matrix = torch.exp(log_ratio) - log_ratio - 1
    sequence_log_ratios = torch.split(k3_matrix, [int(l) for l in response_lengths], dim=0)
    sequence_loss_masks = torch.split(loss_mask, [int(l) for l in response_lengths], dim=0)
    per_seq = [
        masked_mean(sequence_log_ratio, sequence_loss_mask)
        for sequence_log_ratio, sequence_loss_mask in zip(sequence_log_ratios, sequence_loss_masks)
    ]
    metrics["rollout_k3_kl"] = torch.stack(per_seq).sum()

    # Sequence-level perplexity difference metrics
    assert response_lengths is not None and len(response_lengths) > 0
    seq_rollout_means = []
    seq_old_means = []
    start = 0
    for length in response_lengths:
        end = start + int(length)
        mask_chunk = loss_mask[start:end]
        seq_rollout_means.append(masked_mean(rollout_log_prob[start:end], mask_chunk))
        seq_old_means.append(masked_mean(old_log_prob[start:end], mask_chunk))
        start = end
    mean_log_prob_rollout_per_seq = torch.stack(seq_rollout_means)
    mean_log_prob_old_per_seq = torch.stack(seq_old_means)

    diff = mean_log_prob_rollout_per_seq - mean_log_prob_old_per_seq
    # report sums; external reducer divides by num_samples
    metrics["log_ppl_diff"] = diff.sum()
    metrics["log_ppl_abs_diff"] = diff.abs().sum()

    return metrics

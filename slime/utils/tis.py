from typing import Any, Callable, Dict, Optional, Tuple

import torch


def masked_sum(
    tensor: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False
) -> torch.Tensor:
    return (tensor * mask.float()).sum(dim=dim, keepdim=keepdim)


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    total = (tensor * mask.float()).sum(dim=dim, keepdim=keepdim)
    denom = mask.float().sum(dim=dim, keepdim=keepdim)
    return total / (denom + eps)


def calculate_veto_mask(
    log_ratio_for_metrics: torch.Tensor,
    loss_mask: torch.Tensor,
    veto_threshold: Optional[float],
    metrics: Dict[str, Any],
) -> torch.Tensor:
    if veto_threshold is None:
        return torch.ones_like(log_ratio_for_metrics)
    log_veto_threshold = torch.log(torch.tensor(veto_threshold, device=log_ratio_for_metrics.device))
    # For each sequence, check if it has any catastrophic tokens
    catastrophic_tokens = (
        (log_ratio_for_metrics < log_veto_threshold) | (log_ratio_for_metrics > 1 / log_veto_threshold)
    ) & loss_mask.bool()
    has_catastrophic = catastrophic_tokens.any()
    # Create veto mask: 0 if sequence has catastrophic tokens, 1 otherwise
    veto_mask = (~has_catastrophic).float().expand_as(log_ratio_for_metrics)

    # Update metrics
    metrics["catastrophic_ratio"] += masked_mean(has_catastrophic.int(), loss_mask)
    return veto_mask


def truncate(weights: torch.Tensor, loss_mask: torch.Tensor, metrics: Dict[str, Any], *, eps: float) -> torch.Tensor:
    metrics["mean"] += masked_mean(weights, loss_mask)
    metrics["truncate_fraction"] += masked_mean((weights > eps).int(), loss_mask)
    return weights.clamp(0, eps) * loss_mask


def clip(
    weights: torch.Tensor, loss_mask: torch.Tensor, metrics: Dict[str, Any], *, eps_clip: float, eps_clip_high: float
) -> torch.Tensor:
    metrics["mean"] += masked_mean(weights, loss_mask)
    metrics["clip_fraction_low"] += masked_mean((weights < 1 - eps_clip).int(), loss_mask)
    metrics["clip_fraction_high"] += masked_mean((weights > 1 + eps_clip_high).int(), loss_mask)
    return weights.clamp(1 - eps_clip, 1 + eps_clip_high)


def clip_to_zero(
    weights: torch.Tensor, loss_mask: torch.Tensor, metrics: Dict[str, Any], *, eps_clip: float, eps_clip_high: float
) -> torch.Tensor:
    metrics["mean"] += masked_mean(weights, loss_mask)
    metrics["clip_fraction_low"] += masked_mean((weights < 1 - eps_clip).int(), loss_mask)
    metrics["clip_fraction_high"] += masked_mean((weights > 1 + eps_clip_high).int(), loss_mask)
    clip_mask = (weights >= 1 - eps_clip) & (weights <= 1 + eps_clip_high)
    return weights * clip_mask


def compute_train_infer_tis_weights(
    args,
    *,
    new_log_probs: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    response_lengths: Optional[list[int]] = None,
    prefix: str = "",
    tis_function: Callable[[torch.Tensor, torch.Tensor, Dict[str, Any]], torch.Tensor] = None,
) -> Tuple[list[torch.Tensor], Dict[str, Any]]:
    """
    Compute the truncated importance sampling (TIS) weights and metrics.

    Adapted from:

    https://fengyao.notion.site/off-policy-rl#279721e3f6c48092bbe2fcfe0e9c6b33
    https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda

    Args:
        new_log_probs: List of log probs from new policy, one tensor per sequence.
        old_log_probs: List of log probs from old policy, one tensor per sequence.
        - under training/inference tis
            - new_log_probs = training backend
            - old_log_probs = rollout backend
        - under mini batch tis
            - new_log_probs = new batch
            - old_log_probs = old batch
        loss_masks: List of loss masks, one tensor per sequence.
            Note that for single turn RL, the loss_mask is [1] * response_length for each sequence
            For multi turn RL, the tool response will be marked as 0 in the loss_mask.
        response_lengths: The length of the response for each sequence.
        prefix: The prefix for the parameters, indicating which tis is used.

    Returns:
        weights: The importance sampling weights. [batch_size, seq_len]
        metrics: The metrics for the importance sampling weights.
    """

    """
    level: The aggregation level for the importance sampling weights.
        - "token": per-token importance sampling weights, biased low variance.
        - "sequence": product over tokens, unbiased but high variance.
        - "geometric": geometric mean over tokens, biased, medium variance.
    """
    level: str = args.train_infer_tis_level
    metrics: Dict[str, Any] = {}

    # Validate input lists have same length and each sequence has matching shapes
    assert (
        len(old_log_probs) == len(new_log_probs) == len(loss_masks)
    ), f"Input lists must have same length: {len(old_log_probs)} vs {len(new_log_probs)} vs {len(loss_masks)}"

    for i, (old, new, mask) in enumerate(zip(old_log_probs, new_log_probs, loss_masks)):
        assert (
            old.shape == new.shape == mask.shape
        ), f"Sequence {i}: shapes must match - old: {old.shape}, new: {new.shape}, mask: {mask.shape}"

    # TODO: Get device from first tensor and apply to tensors
    # device = old_log_probs[0].device
    SAFETY_BOUND = 20.0
    all_weights = []

    for old_log_prob, new_log_prob, loss_mask in zip(old_log_probs, new_log_probs, loss_masks):
        raw_log_ratio = old_log_prob - new_log_prob

        if level == "token":
            # Token-level IS
            log_ratio_for_metrics = raw_log_ratio
        elif level == "sequence":
            # Sequence-level IS
            agg_log_ratio = masked_sum(raw_log_ratio, loss_mask)
            log_ratio_for_metrics = torch.full_like(raw_log_ratio, agg_log_ratio)
        elif level == "geometric":
            # Geometric mean IS
            agg_log_ratio = masked_mean(raw_log_ratio, loss_mask)
            log_ratio_for_metrics = torch.full_like(raw_log_ratio, agg_log_ratio)
        else:
            raise ValueError(f"Invalid importance sampling level: {level}")

        log_ratio_safe = torch.clamp(log_ratio_for_metrics, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        weights = torch.exp(log_ratio_safe)

        veto_mask = calculate_veto_mask(log_ratio_for_metrics, loss_mask, args.train_infer_tis_veto_threshold, metrics)
        loss_mask = loss_mask & veto_mask  # mask out catastrophic tokens

        weights = tis_function(weights, loss_mask, metrics)

        weights = weights * loss_mask
        weights = weights.detach()

        all_weights.append(weights)

    return weights, metrics


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

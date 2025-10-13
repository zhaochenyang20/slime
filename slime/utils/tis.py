from typing import Any, Callable, Dict, Optional, Tuple

import torch
from slime.backends.megatron_utils.cp_utils import scatter_with_cp


def masked_sum(
    tensor: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False
) -> torch.Tensor:
    return (tensor * mask).sum(dim=dim, keepdim=keepdim)


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    total = (tensor * mask).sum(dim=dim, keepdim=keepdim)
    denom = mask.sum(dim=dim, keepdim=keepdim)
    return total / (denom + eps)


def metrics_add(metrics: Dict[str, Any], key: str, value: float) -> None:
    if key not in metrics:
        metrics[key] = 0
    metrics[key] += value


def metrics_append(metrics: Dict[str, Any], key: str, value: torch.Tensor) -> None:
    if key not in metrics:
        metrics[key] = []
    metrics[key].append(value.clone().detach())


def scatter_cp_and_concat(
    values: list[torch.Tensor], total_lengths: list[int], response_lengths: list[int]
) -> list[torch.Tensor]:
    values = [scatter_with_cp(values[i], total_lengths[i], response_lengths[i]) for i in range(len(values))]
    return torch.cat(values, dim=0)


def calculate_veto_mask(
    log_ratio_for_metrics: torch.Tensor,
    loss_mask: torch.Tensor,
    veto_threshold: Optional[float],
    metrics: Dict[str, Any],
) -> torch.Tensor:
    if veto_threshold is None:
        return torch.ones_like(log_ratio_for_metrics)
    log_veto_threshold = torch.log(torch.tensor(veto_threshold, device=log_ratio_for_metrics.device))
    # For each sequence, if it has any catastrophic tokens, return 0 for the sequence
    catastrophic_tokens = (
        (log_ratio_for_metrics < log_veto_threshold) | (log_ratio_for_metrics > -log_veto_threshold)
    ) & loss_mask.bool()
    has_catastrophic = catastrophic_tokens.any()
    veto_mask = (~has_catastrophic).float().expand_as(log_ratio_for_metrics)

    metrics_append(metrics, "catastrophic_fraction", has_catastrophic.int().expand_as(loss_mask))
    return veto_mask


def truncate(weights: torch.Tensor, loss_mask: torch.Tensor, metrics: Dict[str, Any], *, eps: float) -> torch.Tensor:
    metrics_append(metrics, "truncate_fraction", (weights > eps).int())
    return weights.clamp(0, eps) * loss_mask


def clip(
    weights: torch.Tensor, loss_mask: torch.Tensor, metrics: Dict[str, Any], *, eps_clip: float, eps_clip_high: float
) -> torch.Tensor:
    metrics_append(metrics, "clip_fraction_low", (weights < 1 - eps_clip).int())
    metrics_append(metrics, "clip_fraction_high", (weights > 1 + eps_clip_high).int())
    return weights.clamp(1 - eps_clip, 1 + eps_clip_high) * loss_mask


def clip_to_zero(
    weights: torch.Tensor, loss_mask: torch.Tensor, metrics: Dict[str, Any], *, eps_clip: float, eps_clip_high: float
) -> torch.Tensor:
    metrics_append(metrics, "clip_fraction_low", (weights < 1 - eps_clip).int())
    metrics_append(metrics, "clip_fraction_high", (weights > 1 + eps_clip_high).int())
    clip_mask = (weights >= 1 - eps_clip) & (weights <= 1 + eps_clip_high)
    return weights * clip_mask * loss_mask


def compute_train_infer_tis_weights(
    args,
    *,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    total_lengths: Optional[list[int]] = None,
    response_lengths: Optional[list[int]] = None,
    tis_function: Callable[[torch.Tensor, torch.Tensor, Dict[str, Any]], torch.Tensor],
) -> Tuple[list[torch.Tensor], Dict[str, Any]]:
    """
    Compute the truncated importance sampling (TIS) weights and metrics.

    Adapted from:

    https://fengyao.notion.site/off-policy-rl#279721e3f6c48092bbe2fcfe0e9c6b33
    https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda

    Args:
        train_log_probs: List of log probs from training backend, one tensor per sequence.
        rollout_log_probs: List of log probs from inference backend, one tensor per sequence.
        loss_masks: List of loss masks, one tensor per sequence.
            Note that for single turn RL, the loss_mask is [1] * response_length for each sequence
            For multi turn RL, the tool response will be marked as 0 in the loss_mask.
        response_lengths: The length of the response for each sequence.

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
        len(train_log_probs) == len(rollout_log_probs) == len(loss_masks)
    ), f"Input lists must have same length: {len(train_log_probs)} vs {len(rollout_log_probs)} vs {len(loss_masks)}"

    if total_lengths is not None:
        assert response_lengths is not None, "response_lengths must be provided when total_lengths is set"
        assert len(total_lengths) == len(
            train_log_probs
        ), f"total_lengths must match number of sequences, got {len(total_lengths)} vs {len(train_log_probs)}"

    for i, (train, rollout, mask) in enumerate(zip(train_log_probs, rollout_log_probs, loss_masks)):
        assert (
            train.shape == rollout.shape == mask.shape
        ), f"Sequence {i}: shapes must match - train: {train.shape}, rollout: {rollout.shape}, mask: {mask.shape}"

    # TODO: Get device from first tensor and apply to tensors
    # device = train_log_probs[0].device
    SAFETY_BOUND = 20.0
    all_weights = []

    for train_log_prob, rollout_log_prob, loss_mask in zip(train_log_probs, rollout_log_probs, loss_masks):
        raw_log_ratio = train_log_prob - rollout_log_prob
        loss_mask = loss_mask.float()

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

        # mask out catastrophic tokens
        if args.train_infer_tis_veto_threshold is not None:
            veto_mask = calculate_veto_mask(
                log_ratio_for_metrics, loss_mask, args.train_infer_tis_veto_threshold, metrics
            )

        metrics_append(metrics, "raw_ratio_mean", weights)
        weights = tis_function(weights, loss_mask, metrics)
        metrics_append(metrics, "ratio_mean_after_tis", weights)
        if args.train_infer_tis_veto_threshold is not None:
            weights = weights * veto_mask
            metrics_append(metrics, "ratio_mean_after_veto_mask", weights)

        weights = weights.detach()
        all_weights.append(weights)

    all_weights = scatter_cp_and_concat(all_weights, total_lengths, response_lengths)
    for key, values in metrics.items():
        values = scatter_cp_and_concat(values, total_lengths, response_lengths)
        metrics[key] = values

    return all_weights, metrics

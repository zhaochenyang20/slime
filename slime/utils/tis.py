from typing import Any, Dict, Optional, Tuple

import torch

from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp


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


def metrics_append(metrics: Dict[str, list[torch.Tensor]], key: str, value: torch.Tensor) -> None:
    """
    Any metric should be list[torch.Tensor] with size [num_seq, response_length]
    All tis metrics will be aggregated and averaged by `sum_of_sample_mean` and megatron automatically
    The result will be sequence-level average or token-level if `calculate_per_token_loss` is set.
    You have no need to worry about loss_mask — the sum_of_sample_mean automatically ignores statistics where loss_mask = 0.

    e.g.
    To calculate a token-level metric like the ratio of catastrophic tokens, just append the orignal ratio tensor to the metrics dict.
    To calculate a sequence-level metric like the ratio of vetoed sequences, you should set every value in the tensor to be 0 or 1.
    """
    if key not in metrics:
        metrics[key] = []
    metrics[key].append(value.clone().detach())


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
    catastrophic_tokens = ((log_ratio_for_metrics < log_veto_threshold)) & loss_mask.bool()
    has_catastrophic = catastrophic_tokens.any()
    veto_mask = (~has_catastrophic).float().expand_as(log_ratio_for_metrics)

    # TODO(jiajun): A single catastrophic token may not be enough to veto the entire sequence?
    # May be we can set a threshold for the ratio of catastrophic tokens?
    # If exceeds, veto the entire sequence. If not, only mask the catastrophic tokens.
    metrics_append(metrics, "catastrophic_token_fraction", catastrophic_tokens.int())
    metrics_append(metrics, "catastrophic_seq_fraction", has_catastrophic.int().expand_as(loss_mask))
    return veto_mask


def truncate(
    weights: torch.Tensor, loss_mask: torch.Tensor, metrics: Dict[str, Any], *, upper_bound: float
) -> torch.Tensor:
    assert upper_bound is not None
    metrics_append(metrics, "truncate_fraction", (weights > upper_bound).int())
    return weights.clamp(0, upper_bound) * loss_mask


def clip(
    weights: torch.Tensor, loss_mask: torch.Tensor, metrics: Dict[str, Any], *, lower_bound: float, upper_bound: float
) -> torch.Tensor:
    assert lower_bound is not None and upper_bound is not None and lower_bound < upper_bound
    metrics_append(metrics, "clip_fraction_low", (weights < lower_bound).int())
    metrics_append(metrics, "clip_fraction_high", (weights > upper_bound).int())
    return weights.clamp(lower_bound, upper_bound) * loss_mask


def mask(
    weights: torch.Tensor, loss_mask: torch.Tensor, metrics: Dict[str, Any], *, lower_bound: float, upper_bound: float
) -> torch.Tensor:
    assert lower_bound is not None and upper_bound is not None and lower_bound < upper_bound
    metrics_append(metrics, "mask_fraction_low", (weights < lower_bound).int())
    metrics_append(metrics, "mask_fraction_high", (weights > upper_bound).int())
    mask = (weights >= lower_bound) & (weights <= upper_bound)
    return weights * mask * loss_mask


def compute_train_infer_is_weights(
    args,
    *,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> Tuple[list[torch.Tensor], Dict[str, Any]]:
    """
    Compute the truncated importance sampling (TIS) weights and metrics.
    Args:
        train_log_probs: List of log probs from training backend.
        rollout_log_probs: List of log probs from inference backend.
        loss_masks: List of loss masks.
            Note that for single turn RL, the loss_mask is [1] * response_length for each sequence
            For multi turn RL, the tool response will be marked as 0 in the loss_mask.

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
    level: str = args.train_infer_is_level
    metrics: Dict[str, Any] = {}

    # Validate input lists have same length and each sequence has matching shapes
    assert (
        len(train_log_probs) == len(rollout_log_probs) == len(loss_masks)
    ), f"Input lists must have same length: {len(train_log_probs)} vs {len(rollout_log_probs)} vs {len(loss_masks)}"

    for i, (train, rollout, loss_mask) in enumerate(zip(train_log_probs, rollout_log_probs, loss_masks)):
        assert (
            train.shape == rollout.shape == loss_mask.shape
        ), f"Sequence {i}: shapes must match - train: {train.shape}, rollout: {rollout.shape}, loss_mask: {loss_mask.shape}"

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
        if args.train_infer_is_veto_threshold is not None:
            veto_mask = calculate_veto_mask(
                log_ratio_for_metrics, loss_mask, args.train_infer_is_veto_threshold, metrics
            )

        metrics_append(metrics, "raw_ratio_mean", weights)

        """    
        mode: how to handle the importance sampling weights exceeding the thresholds.
        - "truncated": cap the importance sampling weights at the upper threshold
          https://fengyao.notion.site/off-policy-rl#279721e3f6c48092bbe2fcfe0e9c6b33
        - "mask": zero the importance sampling weights outside the [lower, upper] range.
          https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda
        - "clip": clip the importance sampling weights to the [lower, upper] range.
        """
        if args.train_infer_is_mode == "mask":
            weights = mask(
                weights,
                loss_mask,
                metrics,
                lower_bound=args.train_infer_is_lower_bound,
                upper_bound=args.train_infer_is_upper_bound,
            )
        elif args.train_infer_is_mode == "clip":
            weights = clip(
                weights,
                loss_mask,
                metrics,
                lower_bound=args.train_infer_is_lower_bound,
                upper_bound=args.train_infer_is_upper_bound,
            )
        elif args.train_infer_is_mode == "truncate":
            weights = truncate(weights, loss_mask, metrics, upper_bound=args.train_infer_is_upper_bound)
        else:
            raise ValueError(f"Unsupported train_infer_is_mode: {args.train_infer_is_mode}")

        metrics_append(metrics, "ratio_mean_after_tis", weights)
        if args.train_infer_is_veto_threshold is not None:
            weights = weights * veto_mask
            metrics_append(metrics, "ratio_mean_after_veto_mask", weights)

        weights = weights.detach()
        all_weights.append(weights)

    return all_weights, metrics


def compute_train_infer_is_weights_with_cp(
    args,
    *,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
) -> Tuple[list[torch.Tensor], Dict[str, Any]]:
    """
    Compute the truncated importance sampling (TIS) weights and metrics with context parallel.
    Args:
        train_log_probs: List of log probs from training backend on this cp rank.
        rollout_log_probs: List of log probs from inference backend on this cp rank.
        loss_masks: List of loss masks.
        total_lengths: List of total lengths.
        response_lengths: List of response lengths.
    Returns:
        is_weights: The importance sampling weights. [batch_size, seq_len]
        is_metrics: The metrics for the importance sampling weights.
    """
    # Gather cp slice from other cp ranks
    full_rollout_log_probs = [
        all_gather_with_cp(log_prob, total_length, response_length)
        for log_prob, total_length, response_length in zip(rollout_log_probs, total_lengths, response_lengths)
    ]
    full_old_log_probs = [
        all_gather_with_cp(old_log_prob, total_length, response_length)
        for old_log_prob, total_length, response_length in zip(train_log_probs, total_lengths, response_lengths)
    ]

    # Main logic for is
    is_weights, is_metrics = compute_train_infer_is_weights(
        args=args,
        train_log_probs=full_old_log_probs,
        rollout_log_probs=full_rollout_log_probs,
        loss_masks=loss_masks,
    )

    # Slice cp slice and concat to the full response tensor
    def slice_cp_and_concat(
        values: list[torch.Tensor], total_lengths: list[int], response_lengths: list[int]
    ) -> list[torch.Tensor]:
        # reshape value to the sequence size of the cp rank.
        values = [
            # TODO: A rename of this function ?
            slice_log_prob_with_cp(values[i], total_lengths[i], response_lengths[i])
            for i in range(len(values))
        ]
        return torch.cat(values, dim=0)

    is_weights = slice_cp_and_concat(is_weights, total_lengths, response_lengths)
    for key, values in is_metrics.items():
        values = slice_cp_and_concat(values, total_lengths, response_lengths)
        is_metrics[key] = values

    return is_weights, is_metrics

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    # Optional debug
    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
    # TODO(student): yield RolloutBatch minibatches of size minibatch_size.
    # Requirements:
    # - Let N = batch.input_ids.shape[0] be the number of sampled completions.
    # - If shuffle=True, permute indices with torch.randperm using the provided generator.
    # - Otherwise iterate in the original order 0, 1, ..., N-1.
    # - Slice ALL tensor fields consistently with the same minibatch indices.
    # - Keep task_names / completion_texts aligned with the same indices when present.
    # - If device is not None, move the minibatch to that device before yielding.
    # raise NotImplementedError("student TODO: iter_minibatches")

    N = batch.input_ids.shape[0]

    # Generate indices once for the whole batch
    if shuffle:
        indices = torch.randperm(N, generator=generator, device=batch.input_ids.device)
    else:
        indices = torch.arange(N, device=batch.input_ids.device)

    # Yield minibatches
    for start_idx in range(0, N, minibatch_size):
        end_idx = min(start_idx + minibatch_size, N)
        mb_indices = indices[start_idx:end_idx]

        # Slice all tensor fields natively
        mb_input_ids = batch.input_ids[mb_indices]
        mb_attention_mask = batch.attention_mask[mb_indices]
        mb_completion_mask = batch.completion_mask[mb_indices]
        mb_old_logprobs = batch.old_logprobs[mb_indices]
        mb_ref_logprobs = batch.ref_logprobs[mb_indices]
        mb_rewards = batch.rewards[mb_indices]
        mb_advantages = batch.advantages[mb_indices]

        # Slice optional debug lists if present
        mb_task_names = None
        mb_completion_texts = None
        if batch.task_names is not None or batch.completion_texts is not None:
            mb_indices_list = mb_indices.tolist()
            if batch.task_names is not None:
                mb_task_names = [batch.task_names[i] for i in mb_indices_list]
            if batch.completion_texts is not None:
                mb_completion_texts = [batch.completion_texts[i] for i in mb_indices_list]

        # Construct the minibatch
        mb = RolloutBatch(
            input_ids=mb_input_ids,
            attention_mask=mb_attention_mask,
            completion_mask=mb_completion_mask,
            old_logprobs=mb_old_logprobs,
            ref_logprobs=mb_ref_logprobs,
            rewards=mb_rewards,
            advantages=mb_advantages,
            task_names=mb_task_names,
            completion_texts=mb_completion_texts,
        )

        # Move to device as a single batched operation if requested
        if device is not None:
            mb = mb.to(device)

        yield mb

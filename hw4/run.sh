#!/bin/bash

echo "Submitting Required Run 3: Math Hard + GR-REINFORCE..."
# [cite: 237-265]
uv run modal run --detach scripts/modal_train.py \
  --task math_hard \
  --algo reinforce \
  --output_dir /vol/runs/modal_math_hard_reinforce \
  --steps 201 \
  --batch_size 8 \
  --group_size 8 \
  --min_new_tokens 8 \
  --max_new_tokens 512 \
  --max_prompt_tokens 512 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 3e-5 \
  --minibatch_size 8 \
  --grad_accum_steps 8 \
  --max_grad_norm 0.5 \
  --kl_coef 0.05 \
  --wandb_enabled \
  --wandb_project llm-rl-hw4 \
  --wandb_name math_hard_reinforce \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 8 \
  --cuda_empty_cache_interval 50 \
  --eval_interval 100 \
  --save_interval 100


echo "Submitting Required Run 4: Math Hard + GRPO..."
# [cite: 266-288]
uv run modal run --detach scripts/modal_train.py \
  --task math_hard \
  --algo grpo \
  --output_dir /vol/runs/modal_math_hard_grpo \
  --steps 501 \
  --batch_size 8 \
  --group_size 8 \
  --min_new_tokens 8 \
  --max_new_tokens 512 \
  --max_prompt_tokens 512 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 3e-5 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --grad_accum_steps 8 \
  --clip_eps 0.2 \
  --max_grad_norm 0.5 \
  --kl_coef 0.05 \
  --wandb_enabled \
  --wandb_project llm-rl-hw4 \
  --wandb_name math_hard_grpo \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 8 \
  --cuda_empty_cache_interval 50 \
  --eval_interval 100 \
  --save_interval 100


echo "========================================================"
echo "Submitting Ablation Studies on format_copy + GRPO..."
echo "========================================================"
# The baseline format_copy + GRPO uses:
# --ppo_epochs 2, --kl_coef 0.05, --clip_eps 0.2, --grad_accum_steps 6, --minibatch_size 8

# Ablation 1: Different PPO Epochs (e.g., 4 instead of 2) [cite: 315]
uv run modal run --detach scripts/modal_train.py \
  --task format_copy --algo grpo \
  --output_dir /vol/runs/modal_format_copy_grpo_ppo4 \
  --steps 51 --batch_size 8 --group_size 6 --min_new_tokens 1 --max_new_tokens 24 --lr 3e-5 \
  --ppo_epochs 4 \
  --minibatch_size 8 --grad_accum_steps 6 --clip_eps 0.2 --kl_coef 0.05 --max_grad_norm 0.5 \
  --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name format_copy_grpo_ppo4 \
  --sample_markdown_log_interval 1 --sample_log_interval 10 --sample_log_n 6 --eval_interval 50 --save_interval 50 --warmup_steps 10

# Ablation 2: Smaller KL Coefficient (e.g., 0.01 instead of 0.05) [cite: 315]
uv run modal run --detach scripts/modal_train.py \
  --task format_copy --algo grpo \
  --output_dir /vol/runs/modal_format_copy_grpo_kl0.01 \
  --steps 51 --batch_size 8 --group_size 6 --min_new_tokens 1 --max_new_tokens 24 --lr 3e-5 \
  --ppo_epochs 2 --minibatch_size 8 --grad_accum_steps 6 --clip_eps 0.2 \
  --kl_coef 0.01 \
  --max_grad_norm 0.5 --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name format_copy_grpo_kl0.01 \
  --sample_markdown_log_interval 1 --sample_log_interval 10 --sample_log_n 6 --eval_interval 50 --save_interval 50 --warmup_steps 10

# Ablation 3: Larger KL Coefficient (e.g., 0.2 instead of 0.05) [cite: 315]
uv run modal run --detach scripts/modal_train.py \
  --task format_copy --algo grpo \
  --output_dir /vol/runs/modal_format_copy_grpo_kl0.2 \
  --steps 51 --batch_size 8 --group_size 6 --min_new_tokens 1 --max_new_tokens 24 --lr 3e-5 \
  --ppo_epochs 2 --minibatch_size 8 --grad_accum_steps 6 --clip_eps 0.2 \
  --kl_coef 0.2 \
  --max_grad_norm 0.5 --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name format_copy_grpo_kl0.2 \
  --sample_markdown_log_interval 1 --sample_log_interval 10 --sample_log_n 6 --eval_interval 50 --save_interval 50 --warmup_steps 10

# Ablation 4: Different Clipping Threshold (e.g., 0.1 instead of 0.2) [cite: 315]
uv run modal run --detach scripts/modal_train.py \
  --task format_copy --algo grpo \
  --output_dir /vol/runs/modal_format_copy_grpo_clip0.1 \
  --steps 51 --batch_size 8 --group_size 6 --min_new_tokens 1 --max_new_tokens 24 --lr 3e-5 \
  --ppo_epochs 2 --minibatch_size 8 --grad_accum_steps 6 \
  --clip_eps 0.1 \
  --kl_coef 0.05 --max_grad_norm 0.5 --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name format_copy_grpo_clip0.1 \
  --sample_markdown_log_interval 1 --sample_log_interval 10 --sample_log_n 6 --eval_interval 50 --save_interval 50 --warmup_steps 10

# Ablation 5: grad_accum_steps set to 1 [cite: 316]
uv run modal run --detach scripts/modal_train.py \
  --task format_copy --algo grpo \
  --output_dir /vol/runs/modal_format_copy_grpo_accum1 \
  --steps 51 --batch_size 8 --group_size 6 --min_new_tokens 1 --max_new_tokens 24 --lr 3e-5 \
  --ppo_epochs 2 --minibatch_size 8 \
  --grad_accum_steps 1 \
  --clip_eps 0.2 --kl_coef 0.05 --max_grad_norm 0.5 --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name format_copy_grpo_accum1 \
  --sample_markdown_log_interval 1 --sample_log_interval 10 --sample_log_n 6 --eval_interval 50 --save_interval 50 --warmup_steps 10

echo "All jobs submitted!"

#!/bin/bash

# 32B モデル用 PPO + LoRA シンプル設定（VeRL確実サポート版）
mkdir -p ~/training/ppo_lora_32b_simple
mkdir -p ~/training/ppo_lora_32b_simple/checkpoints
cd ~/training/ppo_lora_32b_simple

# 基本設定
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES
ulimit -v unlimited

# 環境変数
if [ -f .env ]; then
    source .env
    export HF_TOKEN
    export WANDB_API_KEY
fi

export WANDB_PROJECT_NAME="competition_verl_ppo_lora"
export WANDB_RUN_NAME="Qwen3_32b_GRPO_LoRA_simple"

# GRPO + LoRA 確実サポート版
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ppo \
    data.train_files=openai/gsm8k \
    data.val_files=openai/gsm8k \
    data.train_batch_size=64 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$HOME/model/Qwen3_SFT_MATH/checkpoints/global_step_116/huggingface \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    +actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    +actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    +actor_rollout_ref.rollout.n=4 \
    +reward_manager.reward_fn=gsm8k \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$WANDB_PROJECT_NAME \
    trainer.experiment_name=$WANDB_RUN_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.default_local_dir=$HOME/training/ppo_lora_32b_simple/checkpoints \
    trainer.total_epochs=15 2>&1 | tee verl_ppo_simple.log
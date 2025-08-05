#!/bin/bash

# 32B モデル用 PPO + LoRA 設定
mkdir -p ~/training/ppo_lora_32b
mkdir -p ~/training/ppo_lora_32b/checkpoints
cd ~/training/ppo_lora_32b

# 基本的なネットワーク設定
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES
ulimit -v unlimited

# .envファイルから環境変数を読み込み
if [ -f .env ]; then
    source .env
    export HF_TOKEN
    export WANDB_API_KEY
fi

# Wandb設定
export WANDB_PROJECT_NAME="competition_verl_ppo_lora"
export WANDB_RUN_NAME="Qwen3_32b_PPO_LoRA_001"

# PPO + LoRA実行（標準設定）
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=/home/Competition2025/P12/shareP12/models/Qwen3-32B \
    +actor_rollout_ref.model.use_shm=True \
    +actor_rollout_ref.model.lora_rank=32 \
    +actor_rollout_ref.model.lora_alpha=32 \
    +actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    +actor_rollout_ref.rollout.load_format=safetensors \
    +actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=/home/Competition2025/P12/shareP12/models/Qwen3-32B \
    +critic.model.lora_rank=32 \
    +critic.model.lora_alpha=32 \
    +critic.model.enable_gradient_checkpointing=True \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=2 \
    +critic.fsdp_config.param_offload=True \
    +critic.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=2 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$WANDB_PROJECT_NAME \
    trainer.experiment_name=$WANDB_RUN_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.default_local_dir=$HOME/training/ppo_lora_32b/checkpoints \
    trainer.total_epochs=15 2>&1 | tee verl_ppo_lora.log
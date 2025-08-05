#!/bin/bash

# 32B モデル用 PPO + LoRA 極小メモリ設定
mkdir -p ~/training/ppo_lora_32b_minimal
mkdir -p ~/training/ppo_lora_32b_minimal/checkpoints
cd ~/training/ppo_lora_32b_minimal

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
export WANDB_RUN_NAME="Qwen3_32b_PPO_LoRA_minimal"

# PPO + LoRA 極小メモリ設定
HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 \
 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$HOME/model/Qwen3_SFT_MATH/checkpoints/global_step_116/huggingface \
    '+actor_rollout_ref.model.torch_dtype=bfloat16' \
    '+actor_rollout_ref.model.attn_implementation=flash_attention_2' \
    +actor_rollout_ref.model.gradient_checkpointing=true \
    +actor_rollout_ref.model.quantization.load_in_4bit=true \
    +actor_rollout_ref.model.quantization.bnb_4bit_quant_type=nf4 \
    +actor_rollout_ref.model.quantization.bnb_4bit_use_double_quant=true \
    +actor_rollout_ref.model.quantization.bnb_4bit_compute_dtype=bfloat16 \
    +actor_rollout_ref.model.peft.enable_lora=true \
    +actor_rollout_ref.model.peft.r=16 \
    +actor_rollout_ref.model.peft.lora_alpha=32 \
    +actor_rollout_ref.model.peft.lora_dropout=0.05 \
    +actor_rollout_ref.model.peft.target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'] \
    +actor_rollout_ref.actor.optimizer._target_=bitsandbytes.optim.AdamW8bit \
    +actor_rollout_ref.actor.optimizer.lr=1e-6 \
    +actor_rollout_ref.actor.optimizer.bnb_8bit_use_cpu=true \
    +actor_rollout_ref.actor.optimizer.bnb_8bit_use_double_quant=true \
    +actor_rollout_ref.actor.optimizer.bnb_8bit_quant_type=fp8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    '+actor_rollout_ref.rollout.max_model_len=1024' \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    +actor_rollout_ref.rollout.kv_cache_dtype=fp8 \
    +actor_rollout_ref.rollout.block_size=8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=$HOME/model/Qwen3_SFT_MATH/checkpoints/global_step_116/huggingface \
    '+critic.model.torch_dtype=bfloat16' \
    '+critic.model.attn_implementation=flash_attention_2' \
    +critic.model.gradient_checkpointing=true \
    +critic.model.quantization.load_in_4bit=true \
    +critic.model.quantization.bnb_4bit_quant_type=nf4 \
    +critic.model.quantization.bnb_4bit_use_double_quant=true \
    +critic.model.quantization.bnb_4bit_compute_dtype=bfloat16 \
    +critic.model.peft.enable_lora=true \
    +critic.model.peft.r=16 \
    +critic.model.peft.lora_alpha=32 \
    +critic.model.peft.lora_dropout=0.05 \
    +critic.model.peft.target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'] \
    +critic.optimizer._target_=bitsandbytes.optim.AdamW8bit \
    +critic.optimizer.lr=1e-5 \
    +critic.optimizer.bnb_8bit_use_cpu=true \
    +critic.optimizer.bnb_8bit_use_double_quant=true \
    +critic.optimizer.bnb_8bit_quant_type=fp8 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    +critic.fsdp_config.param_offload=true \
    +critic.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    '+reward_manager=naive' \
    trainer.val_before_train=false \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$WANDB_PROJECT_NAME \
    trainer.experiment_name=$WANDB_RUN_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.default_local_dir=$HOME/training/ppo_lora_32b_minimal/checkpoints \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log

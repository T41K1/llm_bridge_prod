mkdir -p ~/training/ppo_001

mkdir -p ~/training/ppo_001/checkpoints

cd ~/training/ppo_001
#基本的なネットワーク設定
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
#CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
#例えば、単一GPUの場合は以下のように設定します：
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#※AMD製のGPUではないため、ROCR_VISIBLE_DEVICES を指定しないようにしてください。指定するとエラーになります。
unset ROCR_VISIBLE_DEVICES
ulimit -v unlimited

export WANDB_PROJECT_NAME="competition_verl_ppo_test"
export WANDB_RUN_NAME="Qwen3_32b_SFT_GRPO_001"


#!/usr/bin/env bash
# ================================================
#  Qwen3-32B で PPO + QLoRA + 8-bit AdamW (CPU オフロード)
# ================================================

################## 0. 依存パッケージ ##################
# CUDA 12.x 環境の例：bitsandbytes-cuda126
#pip install -U bitsandbytes-cuda126>=0.43 \
#               peft>=0.11 \
#               accelerate \
#               transformers>=4.41 \
#               charset-normalizer

################## 1. 環境変数 #######################
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# vLLM と衝突しない断片化対策
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export NVTE_FUSED_ATTN=1
export NVTE_ALLOW_FP8=1      # fp8 KV-cache を使わないならコメントアウト可

################## 2. パス設定 #######################
MODEL_DIR=/home/Competition2025/P12/P12U016/model/Qwen3


PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 data.dataloader_num_workers=0 \
 actor_rollout_ref.model.path=$HOME/model/Qwen3_SFT_MATH/checkpoints/global_step_116/huggingface \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=$HOME/model/Qwen3_SFT_MATH/checkpoints/global_step_116/huggingface \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console'] \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.default_local_dir=$HOME/training/ppo/checkpoints \
 trainer.logger=['console','wandb'] \
 trainer.project_name=$WANDB_PROJECT_NAME \
 trainer.experiment_name=$WANDB_RUN_NAME \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log
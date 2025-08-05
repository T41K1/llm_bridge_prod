#!/bin/bash
#SBATCH --job-name=grpo_full_pipeline
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gres=gpu:8 # GPUが必要な場合
#SBATCH --nodelist=osk-gpu[85]
#SBATCH --cpus-per-task=240
#SBATCH --time=50:00:00

nvidia-smi
# エラー時に停止
set -e

# .envファイルの読み込み
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "Warning: .env file not found at $ENV_FILE"
    echo "Please copy .env.example to .env and configure your tokens"
    exit 1
fi

echo "=== GRPO Full Pipeline Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Repository root: $REPO_ROOT"

# Step 1-0: Python仮想環境の起動
echo "=== Step 1-0: Setting up Python environment ==="
module reset
module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311

source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# condaコマンドが使えることを確認
which conda && echo "====" && conda --version

# CONDA_PATHを.envファイルから読み込み（デフォルト値を設定）
export CONDA_PATH="${CONDA_PATH:-~/conda_env}"
source ~/.bashrc
conda init
conda config --set auto_activate_base false

# Python仮想環境のリセットと有効化
conda deactivate || true
conda deactivate || true
conda activate $CONDA_PATH

echo "Python environment activated"

# Step 1-1: 認証の自動化
echo "=== Step 1-1: Authentication ==="

# HuggingFace自動ログイン（環境変数から）
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "Error: HUGGING_FACE_HUB_TOKEN environment variable is not set"
    echo "Please set it in your ~/.bashrc or job script"
    exit 1
fi

# HuggingFaceトークンファイルに書き込み（非対話的ログイン）
mkdir -p ~/.cache/huggingface
echo "$HUGGING_FACE_HUB_TOKEN" > ~/.cache/huggingface/token
echo "HuggingFace authentication completed"

# Wandb認証（手動で事前にログインしてください）
wandb login || {
    echo "Error: Please run 'wandb login' manually before starting this job"
    echo "You can do this by running 'wandb login' in your terminal"
    exit 1
}

# Step 1-4: 強化学習（GRPO）の実行
echo "=== Step 1-4: GRPO Training ==="

# ディレクトリ作成
mkdir -p ~/training/grpo_00
mkdir -p ~/training/grpo/checkpoints
cd ~/training/grpo

# 基本的なネットワーク設定
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES
ulimit -v unlimited

# Wandb設定（.envファイルから読み込み、デフォルト値を設定）
export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-competition_verl_test}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-Qwen3_32b_SFT_GRPO_$(date +%Y%m%d_%H%M%S)}"



echo "Starting GRPO training..."

# GRPO学習実行
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=2048 \
 data.max_response_length=14336 \
 data.dataloader_num_workers=0 \
 actor_rollout_ref.model.path=$HOME/model/Qwen3_SFT_MATH/checkpoints/global_step_116/huggingface \
 actor_rollout_ref.actor.optim.lr=5e-7 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
 actor_rollout_ref.rollout.n=4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
 algorithm.adv_estimator=grpo \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console'] \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.default_local_dir=$HOME/training/grpo_001/checkpoints \
 trainer.logger=['console','wandb'] \
 trainer.project_name=$WANDB_PROJECT_NAME \
 trainer.experiment_name=$WANDB_RUN_NAME \
 trainer.total_epochs=15 2>&1 | tee verl_grpo.log

echo "GRPO training completed"

# Step 1-5: チェックポイントの変換
echo "=== Step 1-5: Converting checkpoint to HuggingFace format ==="

# 最新のチェックポイントを探す
LATEST_CHECKPOINT=$(find $HOME/training/grpo_001/checkpoints -name "global_step_*" -type d | sort -V | tail -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found!"
    exit 1
fi

echo "Converting checkpoint: $LATEST_CHECKPOINT"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $LATEST_CHECKPOINT/actor \
    --target_dir $LATEST_CHECKPOINT/actor/huggingface

echo "Checkpoint conversion completed"

# Step 1-6: モデルのアップロード（オプション）
echo "=== Step 1-6: Model upload (optional) ==="

if [ -n "$HUGGING_FACE_HUB_TOKEN" ] && [ -n "$YOU_HF_TEAM" ] && [ -n "$YOU_HF_PROJECT_NAME" ]; then
    echo "Uploading model to HuggingFace Hub..."
    huggingface-cli upload \
        $YOU_HF_TEAM/Qwen3-32B-SFT-MATH \
        $LATEST_CHECKPOINT/actor/huggingface \
        --token $HUGGING_FACE_HUB_TOKEN
    echo "Model upload completed"
else
    echo "Skipping model upload (HuggingFace variables not set)"
fi

echo "=== GRPO Full Pipeline Completed ==="
echo "End time: $(date)"
echo "Checkpoint location: $LATEST_CHECKPOINT/actor/huggingface"
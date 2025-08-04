mkdir -p ~/training/grpo_004

mkdir -p ~/training/grpo_004/checkpoints

cd ~/training/grpo_004
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


export WANDB_PROJECT_NAME="competition_verl_test"
export WANDB_RUN_NAME="Qwen3_32b_SFT_GRPO_004"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
   data.train_files=$HOME/data/gsm8k/train.parquet \
   data.val_files=$HOME/data/gsm8k/test.parquet \
   data.train_batch_size=256 \
   data.max_prompt_length=512 \
   data.max_response_length=1024 \
   data.dataloader_num_workers=0 \
   actor_rollout_ref.model.path=/home/Competition2025/P12/shareP12/models/Qwen3-32B \
   actor_rollout_ref.actor.optim.lr=3e-7 \
   actor_rollout_ref.actor.ppo_mini_batch_size=32 \
   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
   actor_rollout_ref.actor.use_kl_loss=True \
   actor_rollout_ref.actor.kl_loss_coef=0.001 \
   +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
   actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
   actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
   actor_rollout_ref.rollout.n=4 \
   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
   algorithm.adv_estimator=grpo \
   trainer.n_gpus_per_node=8 \
   trainer.nnodes=1 \
   trainer.save_freq=10 \
   trainer.test_freq=10 \
   trainer.default_local_dir=$HOME/training/grpo_003/checkpoints \
   trainer.logger=['console','wandb'] \
   trainer.project_name=$WANDB_PROJECT_NAME \
   trainer.experiment_name=$WANDB_RUN_NAME \
   trainer.total_epochs=15 2>&1 | tee verl_grpo.log
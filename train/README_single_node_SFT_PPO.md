# Train

## 前提

* 計算環境:  1 node, 8 GPU (Nvidia H100)
  * YOU_TEAM を案内された partition 番号に置き換えてください。
  * 例: `$ srun --partition=YOU_TEAM --nodes=1 --gpus-per-node=8 --cpus-per-task=240 --time=30:30:00 --nodelist=osk-gpu[YOU_TEAM] --job-name="SFT_PPO_test" --pty bash -i`



## Step 1. シングルードモデルのファインチューニング

### Step 1-0.  Python仮想環境の起動

``` sh
# 現在のモジュール環境をリセットする（読み込まれている全てのモジュールをアンロード）
module reset

# NCCL（NVIDIA Collective Communications Library）バージョン2.22.3を読み込む
module load nccl/2.22.3

# HPC-X（高性能通信ライブラリ）バージョン2.18.1をCUDA 12およびGCCに対応する構成で読み込む
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

module load miniconda/24.7.1-py311

source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# condaコマンドが使えることを確認。
which conda && echo "====" && conda --version

#step0 でインストールした conda のディレクトリ
export CONDA_PATH="~/conda_env"

source ~/.bashrc

conda init

conda config --set auto_activate_base false

# 念のため既に有効化されているPython仮想環境がある場合に備えてリセットのために無効化する。
conda deactivate
conda deactivate

# 作成したPython仮想環境を有効化。
conda activate $CONDA_PATH

```

### Step 1-1. シングルード学習を行うための下準備
``` sh
# 事前に wandb と huggingface のアカウントを準備しておいてください。
# wandb と huggingface にログインしてください。
# huggingfaceのアクセストークンは以下のページから発行できます: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
huggingface-cli login
#wandb は自動的にアクセストークン入力用のURLを表示します。
wandb login
```

### Step 1-4. 強化学習PPOの実行

rollout の設定に以下を追加してください：

```sh
actor_rollout_ref.rollout.tensor_model_parallel_size=<GPU 数>
```

例えば、GPU が 1 枚で推論の場合は `1` と指定します。

``` sh
mkdir -p ~/training/ppo

mkdir -p ~/training/ppo/checkpoints

cd ~/training/ppo
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

#YOU_TEAM_ENTITY_NAME を wandb の組織名に置き換えてください。
# export WANDB_ENTITY="YOU_TEAM_ENTITY_NAME"
export WANDB_PROJECT_NAME="competition_verl_test_ppo"
export WANDB_RUN_NAME="Qwen3-32b_sft_PPO_test"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 data.dataloader_num_workers=0 \
 actor_rollout_ref.model.path=$HOME/model/Qwen3_SFT_MATH/checkpoints/global_step_116/huggingface \
 actor_rollout_ref.actor.optim.lr=3e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
 +actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
 +actor_rollout_ref.rollout.vllm_dtype=bf16 \
 +critic.fsdp_config.model_dtype=bf16 \
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


```
学習済みモデルのパスは以下の通りです。
しかし、huggingfaceのHF形式ではないため、さらに変換が必要です。
```sh
cd $HOME/training/ppo/checkpoints/global_step_435
ls -lh
```

### Step 1-5. 強化学習PPOのチェックポイントの変換

保存されたチェックポイントは、verl.model_merger モジュールを使用して Huggingface モデルにマージできます。例えば、次のようにします：
```sh
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $HOME/training/ppo/checkpoints/global_step_435/actor \
    --target_dir $HOME/training/ppo/checkpoints/global_step_435/actor/huggingface
```

### Step 1-6. ファインチューニング済みモデルのアップロード
YOU_HF_TEAM と YOU_HF_TOKEN を huggingface のチーム名とアクセストークンに、YOU_HF_PROJECT_NAME をプロジェクト名に置き換えてください。

```sh
# アップロードスクリプトを実行。
python $HOME/llm_bridge_prod/train/scripts/upload_tokenizer_and_finetuned_model_to_huggingface_hub.py \
    --input_tokenizer_and_model_dir $HOME/training/ppo/checkpoints/global_step_435/actor/huggingface \
    --hf_token $YOU_HF_TOKEN \
    --repo_id $YOU_HF_TEAM/$YOU_HF_PROJECT_NAME
```
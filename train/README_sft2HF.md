### SFTをした後に、fsdpからhf形式にモデルを変換する

元のREADMEは、PPOをした後のものを使用していたので、念のためにSFT後の方法も記載する


今回は、個人のディレクトリにある想定でやっているので、パスは適宜変更してください
#### 1. SFT済みチェックポイントの確認
```sh
# SFT完了後のチェックポイント確認
cd $HOME/model/Qwen3_SFT_MATH/checkpoints
ls -la

# 対象のチェックポイントを確認
cd global_step_116
ls -lh
```

#### 2. HuggingFace形式への変換
```sh
# SFT済みチェックポイントをHF形式に変換
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $HOME/model/Qwen3_SFT_MATH/checkpoints/global_step_116 \
    --target_dir $HOME/model/Qwen3_SFT_MATH/checkpoints/global_step_116/huggingface
```

#### 3. 変換確認
```sh
# 変換されたHF形式モデルの確認
cd $HOME/model/Qwen3_SFT_MATH/checkpoints/global_step_116/huggingface
ls -lh

# 以下のファイルがあることを確認:
# - config.json
# - pytorch_model.bin (または model.safetensors)
# - tokenizer.json
# - tokenizer_config.json
# - special_tokens_map.json
```


#### 4. HuggingFaceへのアップロード
YOU_HF_TEAM と YOU_HF_TOKEN を huggingface のチーム名とアクセストークンに、YOU_HF_PROJECT_NAME をプロジェクト名に置き換えてください。

```sh
# SFT済みモデルをHuggingFaceにアップロード
python $HOME/llm_bridge_prod/train/scripts/upload_tokenizer_and_finetuned_model_to_huggingface_hub.py \
    --input_tokenizer_and_model_dir $HOME/model/Qwen3_SFT_MATH/checkpoints/global_step_116/huggingface \
    --hf_token $HF_TOKEN \
    --repo_id Ta1k1/Qwen3-32B-SFT-MATH
```

上のがうまくいかないので、huggingface-cliを使用する

```sh
# HuggingFace CLIでアップロード
huggingface-cli upload \
    Ta1k1/Qwen3-32B-SFT-MATH \
    $HOME/model/Qwen3_SFT_MATH/checkpoints/global_step_116/huggingface \
    --token $HF_TOKEN
```

こんな感じ
一列目は自分のアカウント名/モデルアップロード名
二列目は自分がアップロードするファイル
三列目はHFのTOKENを設定
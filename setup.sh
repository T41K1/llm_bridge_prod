#!/bin/bash

# GRPO Training Environment Setup Script
# このスクリプトはリポジトリをクローンした後に実行してください

echo "=== GRPO Training Environment Setup ==="

# .envファイルの作成
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ .env file created from .env.example"
        echo ""
        echo "IMPORTANT: Please edit .env file and set your HuggingFace token:"
        echo "  - HUGGING_FACE_HUB_TOKEN: Get from https://huggingface.co/settings/tokens"
        echo "  - Update other settings as needed"
        echo ""
    else
        echo "❌ .env.example not found"
        exit 1
    fi
else
    echo "⚠️  .env file already exists"
fi

# スクリプトに実行権限を付与
echo "Setting executable permissions on scripts..."
find train/scripts -name "*.sh" -exec chmod +x {} \;
echo "✓ Executable permissions set"

# ディレクトリ構造の確認
echo ""
echo "Repository structure:"
echo "├── .env                           # Your tokens (edit this!)"
echo "├── .env.example                   # Template file"
echo "├── setup.sh                      # This setup script"
echo "└── train/scripts/single_node_PPO/"
echo "    └── sbatch_full_grpo_pipeline.sh  # Main SLURM script"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your HuggingFace token"
echo "2. Login to wandb manually: wandb login"
echo "3. Verify paths in .env match your environment"
echo "4. Submit job: sbatch train/scripts/single_node_PPO/sbatch_full_grpo_pipeline.sh"
echo ""
echo "For help: cat train/scripts/single_node_PPO/README.md"
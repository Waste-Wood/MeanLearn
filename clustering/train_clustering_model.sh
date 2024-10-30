#!/usr/bin/zsh
#SBATCH -J train
#SBATCH -o ./logger/train.txt
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:a100-sxm-64gb:1


source ~/.zshrc
conda activate lora

python train_clustering_model.py

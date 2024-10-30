#!/usr/bin/zsh
#SBATCH -J clustering_roberta_large
#SBATCH -o ./logger/clustering_roberta_large.txt
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1

source ~/.zshrc
conda activate lora

python clustering.py --model ./model --input_dir ../data/arc --output_dir ./arc --thre 0.6 --batch_size 128
python clustering.py --model ./model --input_dir ../data/bbh/classification --output_dir ./bbh/classification --thre 0.6 --batch_size 128
python clustering.py --model ./model --input_dir ../data/agieval --output_dir ./agieval --thre 0.6 --batch_size 128
python clustering.py --model ./model --input_dir ../data/commonsense --output_dir ./commonsense --thre 0.6 --batch_size 128
python clustering.py --model ./model --input_dir ../data/mmlu --output_dir ./mmlu --thre 0.6 --batch_size 128
python clustering.py --model ./model --input_dir ../data/race --output_dir ./race --thre 0.6 --batch_size 128







# train MeanLearn-13B basd on Orca-2-13B
CUDA_VISIBLE_DEVICES=0,1,2 python train.py \
  --data "./data/gpt-4/ablation/train.jsonl" \
  --base_model "/home/kxiong/LLMs/Orca-2/13B" \
  --dev "./data/gpt-4/dev.jsonl" \
  --output_dir "./lora/13B/epoch1_5e-5" \
  --batch_size 240 \
  --micro_batch_size 80 \
  --epochs 1 \
  --lr 5e-5




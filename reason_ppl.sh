# [agieval]
# MeanLearn-7B
python reason_instant.py \
 --input_dir "./data/agieval" \
 --lora_dir "./lora/7B/epoch1_5e-5" \
 --output_dir "./output/agieval/7B/lora/epoch1_5e-5" \
 --lora True

# Orca-2
python reason_instant.py \
 --input_dir "./data/agieval" \
 --base_model "path to Orca-2-7B" \
 --output_dir "./output/agieval/7B/orca2"



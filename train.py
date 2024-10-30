import os
import torch
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    )
import json
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer
from argparse import ArgumentParser


train_on_inputs = False
def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
            prompt,
            truncation=True,
            padding=False,
            return_tensors=None,
            max_length=512
        )
    if result["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    system = data_point["system"]

    full_prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{data_point['input']}<|im_end|>\n<|im_start|>assistant\n{data_point['output']}<|im_end|>"

    tokenized_full_prompt = tokenize(full_prompt)

    if not train_on_inputs:
        user_prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{data_point['input']}<|im_end|>"
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]
    return tokenized_full_prompt


parser = ArgumentParser()

parser.add_argument('--data', type=str, default='./data/ecare/full_dataset.jsonl')
parser.add_argument('--dev', type=str, default="./")
parser.add_argument('--base_model', type=str, default='/ssd1/huggingface_transformers/llama/hf_weights/7B_chat')
parser.add_argument('--output_dir', type=str, default='./lora/7B')

parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--micro_batch_size', type=int, default=24)
parser.add_argument('--val_set_size', type=int, default=0)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-4)

parser.add_argument('--lora_r', type=int, default=8)
parser.add_argument('--lora_alpha', type=int, default=16)
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--lora_target_modules', type=list, default=["q_proj", "v_proj"])

parser.add_argument('--group_by_length', type=bool, default=True)

opt = parser.parse_args()


if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {opt.base_model}\n"
        f"data_path: {opt.data}\n"
        f"output_dir: {opt.output_dir}\n"
        f"batch_size: {opt.batch_size}\n"
        f"micro_batch_size: {opt.micro_batch_size}\n"
        f"num_epochs: {opt.epochs}\n"
        f"learning_rate: {opt.lr}\n"
        f"val_set_size: {opt.val_set_size}\n"
        f"lora_r: {opt.lora_r}\n"
        f"lora_alpha: {opt.lora_alpha}\n"
        f"lora_dropout: {opt.lora_dropout}\n"
        f"lora_target_modules: {opt.lora_target_modules}\n"
        f"group_by_length: {opt.group_by_length}\n"
    )

assert (
    opt.base_model
), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

gradient_accumulation_steps = opt.batch_size // opt.micro_batch_size
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1

if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = gradient_accumulation_steps // world_size

tokenizer = LlamaTokenizer.from_pretrained(opt.base_model, legacy=True)

model = LlamaForCausalLM.from_pretrained(
    opt.base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map=device_map
)

tokenizer.pad_token_id = (0)
tokenizer.padding_side = 'left'

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=opt.lora_r,
    lora_alpha=opt.lora_alpha,
    target_modules=opt.lora_target_modules,
    lora_dropout=opt.lora_dropout,
    bias='none',
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

data = load_dataset("json", data_files=opt.data)
if len(opt.dev) > 0:
    data_eval = load_dataset("json", data_files=opt.dev)
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = data_eval["train"].shuffle().map(generate_and_tokenize_prompt)
else:
    train_val = data["train"].train_test_split(test_size=opt.val_set_size, shuffle=True, seed=42)
    train_data = (train_val["train"].shuffle().map(generate_and_tokenize_prompt))
    val_data = (train_val["test"].shuffle().map(generate_and_tokenize_prompt))
model.print_trainable_parameters()


if not ddp and torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

os.environ["WANDB_DISABLED"] = "true"
trainer = transformers.Trainer(
    model=model,
    eval_dataset=val_data,
    train_dataset=train_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=opt.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.1,
        num_train_epochs=opt.epochs,
        fp16=True,
        logging_steps=8,
        optim='adamw_torch',
        learning_rate=opt.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        eval_steps=100,
        output_dir=opt.output_dir,
        save_total_limit=3,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=opt.group_by_length,
        report_to="none"
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=2, return_tensors="pt", padding=True
    ),
)


model.config.use_cache = False

trainer.train()

model.save_pretrained(opt.output_dir, safe_serialization=False)
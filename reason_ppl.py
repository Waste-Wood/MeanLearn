from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from argparse import ArgumentParser
import torch
from tqdm import tqdm
import os
from tools import DynamicDataset
from torch.utils.data import DataLoader
import json
import pdb


def process_input(data):
    systems, users, assistants = [], [], []
    for key in tqdm(data):
        for sub_key in data[key]:
            if "label" not in sub_key:
                continue
            if data[key][sub_key]['testing input'].count("\n\n") > 1:
                temp_parts = data[key][sub_key]['testing input'].strip().split("\n\n")
                system = temp_parts[0]
                temp = "\n\n".join(temp_parts[1:]).replace("\n\n\n", "\n").replace("\n\n", "\n")
            elif data[key][sub_key]['testing input'].count("\n\n") == 1:
                system, temp = data[key][sub_key]['testing input'].strip().split("\n\n")
            else:
                system = "You are a cautious assistant."
                temp = data[key][sub_key]['testing input']

            parts = temp.split("\n")
            assistants.append(parts[-1])
            users.append("\n".join(parts[:-1]))
            systems.append(system)
    return systems, users, assistants


def generate_and_tokenize_prompt(hps, syms, usrs, assts):
    if hps.lora or 'orca' in hps.base_model:
        full_prompts = [f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{ipt}<|im_end|>\n<|im_start|>assistant\n{opt}<|im_end|>" for system, ipt, opt in zip(syms, usrs, assts)]
    elif "Vicuna" in hps.base_model:
        full_prompts = [f"SYSTEM: {system} USER: {ipt} ASSISTANT: {opt}" for system, ipt, opt in zip(syms, usrs, assts)]
    elif "WizardLM" in hps.base_model:
        full_prompts = [f"{system}\n{ipt}\n\n### Response:\n{opt}" for system, ipt, opt in zip(syms, usrs, assts)]
    else:
        full_prompts = [f"{system}\n\n{ipt}\n{opt}" for system, ipt, opt in zip(syms, usrs, assts)]
    
    tokenized_full_prompt = tokenizer(full_prompts, padding=True, return_tensors='pt')
    input_ids = tokenized_full_prompt.input_ids
    attention_mask = tokenized_full_prompt.attention_mask
    labels = torch.where(attention_mask == 0, -100, input_ids)

    return input_ids.cuda(), attention_mask.cuda(), labels.cuda()


def compute_loss(labels, logits):
    _, _, vocab = logits.shape
    loss = []
    for logit, label in zip(logits, labels):
        shift_logit = logit[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()

        shift_logit = shift_logit.view(-1, vocab)
        shift_label = shift_label.view(-1)

        shift_label = shift_label.to(shift_logit.device)
        tmp_loss = loss_fct(shift_logit, shift_label).item()
        loss.append(tmp_loss)
    return loss


parser = ArgumentParser()

parser.add_argument('--input_dir', type=str, default='./data/gpt-4')
parser.add_argument('--base_model', type=str, default='/home/kxiong/LLMs/Orca-2/7B')
parser.add_argument('--lora_dir', type=str, default='./lora/gpt-4')
parser.add_argument('--output_dir', type=str, default='./output/')
parser.add_argument("--batch_size", type=int, default=80)
parser.add_argument("--data_name", type=str, default="test")
parser.add_argument("--lora", type=bool, default=False)

opt = parser.parse_args()

print(opt)

tokenizer = LlamaTokenizer.from_pretrained(opt.base_model)
tokenizer.pad_token_id = (0)
tokenizer.padding_side = 'left'

if opt.lora:
    model = LlamaForCausalLM.from_pretrained(
        opt.lora_dir,
        load_in_8bit=False,
        device_map="auto",
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        opt.base_model,
        load_in_8bit=False,
        device_map="auto",
    )

finished = []
model.eval()
files = [f for f in os.listdir(opt.input_dir) if f.endswith(".json")]

for f in files:
    opt.data_name = f.split(".")[0]

    if os.path.exists(f"{opt.output_dir}/{f}"):
        print(f"Skipping {opt.data_name}...")
        continue
    
    if opt.data_name in [
        "lukaemon_mmlu_high_school_european_history",
        "lukaemon_mmlu_professional_law_0",
        "lukaemon_mmlu_high_school_us_history",
        "lukaemon_mmlu_high_school_world_history",
        "lukaemon_mmlu_international_law",
        "lukaemon_mmlu_professional_law_1",
        "lukaemon_mmlu_world_religions",
        "lukaemon_mmlu_prehistory",
        "lukaemon_mmlu_college_medicine"
    ] or "agieval" in opt.data_name or "race" in opt.data_name:
        opt.batch_size = 10
    else:
        opt.batch_size = 32

    if opt.data_name in finished:
        print(f"Skipping {opt.data_name}...")
        continue
    else:
        print(f"Processing {opt.data_name}...")
        print(f"Batch Size {opt.batch_size}...")
    samples = json.load(open(f"{opt.input_dir}/{opt.data_name}.json", "r"))
    data = process_input(samples)

    dataset = DynamicDataset(*data)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    all_loss = []
    
    for batch in tqdm(loader):
        systems, users, assistants = batch
        
        input_ids, attention_mask, labels = generate_and_tokenize_prompt(opt, systems, users, assistants)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            ).logits
        all_loss += compute_loss(labels, logits)

    fo = open(f"{opt.output_dir}/{opt.data_name}.json", 'w')
    index = 0
    for key in samples:
        for sub_key in samples[key]:
            if "label" not in sub_key:
                continue
            samples[key][sub_key]['Loss'] = all_loss[index]
            index += 1
    json.dump(samples, fo, indent=4)

    fo.close()







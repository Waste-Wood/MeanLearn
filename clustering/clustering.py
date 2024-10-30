from transformers import AutoTokenizer, AutoModel
import json
from argparse import ArgumentParser
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import pdb


def process_instance(hps, instance):
    inputs = instance["label: A"]["testing input"]
    if "agieval" in hps.input_dir:
        question = inputs.split("\n")[1:-1]
        question = "\n".join(question)
    elif any(keyword in hps.input_dir for keyword in ["arc", "mmlu", "race"]):
        question = inputs.split("\n")[:-1]
        question = "\n".join(question)
    else:
        question = inputs.split("\n\n")[1:]
        question = "\n".join(question).split('\n')[:-1]
        question = "\n".join(question)
    return question


@torch.no_grad()
def find_proper_index(scores, thre, clusters):
    sorted_socres, indices = torch.sort(scores, descending=True)
    sorted_socres = sorted_socres.cpu().tolist()
    indices = indices.cpu().tolist()
    flag = False
    for s, i in zip(sorted_socres, indices):
        if s >= thre and len(clusters[i]) < 3:
            flag = True
            break
        else:
            continue
    return flag, i


@torch.no_grad()
def clustering(hps, centers, cluster_tensors, clusters, samples, index):
    # centers: n x dim
    # samples: bsz x dim
    # pdb.set_trace()
    for sample in samples:
        # scores = sample @ centers.t() # n
        scores = torch.cosine_similarity(sample.repeat(centers.size(0), 1), centers)
        flag, max_idx = find_proper_index(scores, hps.thre, clusters)
        if flag:
            clusters[max_idx].append(index)
            cluster_tensors[max_idx] = torch.cat([cluster_tensors[max_idx], sample.unsqueeze(0)], dim=0)
            centers[max_idx] = torch.mean(cluster_tensors[max_idx], dim=0)
            index += 1
        else:
            size = len(clusters)
            clusters[size] = [index]
            cluster_tensors[size] = sample.unsqueeze(0)
            centers = torch.cat([centers, sample.unsqueeze(0)], dim=0)
            index += 1
    return centers, cluster_tensors, clusters, index



@torch.no_grad()
def process_file(args, f):
    print(f"Processing file: {f}...")

    data = json.load(open(os.path.join(args.input_dir, f), "r"))
    instances = [process_instance(args, data[key]) for key in data]
    tokens = tokenizer(instances, padding=True, truncation=True, return_tensors="pt", max_length=512)

    DATA = TensorDataset(tokens["input_ids"], tokens["attention_mask"])
    loader = DataLoader(DATA, batch_size=args.batch_size, shuffle=False)

    cluster_center = None
    clusters = {}
    cluster_tensors = {}
    index = 0
    for batch in tqdm(loader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, attention_mask = batch
        outputs = model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        # outputs = torch.softmax(outputs, dim=-1)

        if cluster_center is None:
            cluster_center = outputs[0].unsqueeze(0)
            clusters[0] = [0]
            cluster_tensors[0] = outputs[0].unsqueeze(0)
            index += 1
            cluster_center, cluster_tensors, clusters, index = clustering(args, cluster_center, cluster_tensors, clusters, outputs[1:, :], index)
        else:
            cluster_center, cluster_tensors, clusters, index = clustering(args, cluster_center, cluster_tensors, clusters, outputs, index)

    return clusters


parser = ArgumentParser()
parser.add_argument("--model", type=str, default="./model")
parser.add_argument("--input_dir", type=str, default="../data/arc")
parser.add_argument("--thre", type=float, default=0.8)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained("/home/kxiong/huggingface_transformers/roberta-large")
model = AutoModel.from_pretrained(args.model, device_map="auto")
model.eval()

files = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
model_name = args.model.split("/")[-1]

for f in files:
    clusters = process_file(args, f)
    json.dump(clusters, open(os.path.join(args.output_dir, f"{model_name}_{f}"), "w"), indent=4)


















import json
from argparse import ArgumentParser
import os


parser = ArgumentParser()
parser.add_argument("--data", type=str, default="arc")
parser.add_argument("--model", type=str, default="roberta-large")
parser.add_argument("--input_dir", type=str, default="llama-2")
opt = parser.parse_args()

base_index = ord("A")
opt.data = opt.input_dir.split("/")[2]

if "bbh" not in opt.data:
    files = [f for f in os.listdir(f"{opt.input_dir}") if f.endswith(".json")]
else:
    files = [f for f in os.listdir(f"{opt.input_dir}/classification") if f.endswith(".json")]

hards, softs = [], []

print(f"{'-'*20} {opt.data} {'-'*20}")

for f in files:
    if "bbh" not in opt.data:
        clustering_file = f"./clustering/{opt.data}/{opt.model}_{f}"
        prediction_file = os.path.join(opt.input_dir, f)
    else:
        clustering_file = f"./clustering/{opt.data}/classification/{opt.model}_{f}"
        prediction_file = os.path.join(opt.input_dir, "classification", f)

    clusters = json.load(open(clustering_file, "r"))
    data = json.load(open(prediction_file, "r"))

    count_hard, count_soft = 0, 0
    for key in clusters:
        cluster_instances = [data[str(idx)] for idx in clusters[key]]
        temp_count = 0
        for cluster_instance in cluster_instances:
            losses = [cluster_instance[d]['Loss'] for d in cluster_instance if "label" in d]
            if "bbh" in opt.data:
                label = cluster_instance['gold'][1]
            else:
                label = cluster_instance['gold']
            prediction = chr(base_index + losses.index(min(losses)))

            if prediction == label:
                temp_count += 1
        if temp_count == len(cluster_instances):
            count_hard += 1

    hards.append(count_hard / len(clusters))

print(f"[AbsAcc] Overall: {sum(hards) / len(hards)}")









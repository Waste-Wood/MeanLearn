import json
import os
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--input_dir", type=str, default="")
opt = parser.parse_args()

files = [f for f in os.listdir(opt.input_dir) if f.endswith(".json")]

base_index = ord("A")

print(f"--------------------{opt.input_dir}-----------------------")
lengths = []
counts = []

for f in files:
    data = json.load(open(os.path.join(opt.input_dir, f), "r"))
    fo = open(os.path.join(opt.input_dir, f), "w")
    count = 0
    lengths.append(len(data))
    for key in data:
        losses = [data[key][d]['Loss'] for d in data[key] if "label" in d]
        label = data[key]['gold']
        prediction = chr(base_index + losses.index(min(losses)))

        if label == prediction:
            count += 1
        data[key]['prediction'] = prediction
    counts.append(count)
    json.dump(data, fo, indent=4)

if "mmlu" in opt.input_dir:
    print(f"[MMLU]:\t{sum(counts) / sum(lengths)}")
else:
    print(f"[TOTAL]:\t{sum(counts) / sum(lengths)}")






















import json
import jsonlines
from argparse import ArgumentParser
import os


parser = ArgumentParser()
parser.add_argument("--input_dir", type=str, default="")
opt = parser.parse_args()

original_data = [d for d in jsonlines.open("./data/absr/test_ppl.jsonl", "r")]
data = json.load(open(os.path.join(opt.input_dir, "test_instant.json"), "r"))

abstract_dict = {}
for i, d in enumerate(original_data):
    if d['fact'] in abstract_dict:
        abstract_dict[d['fact']].append(i)
    else:
        abstract_dict[d['fact']] = [i]

prediction_dict = {}
for key in abstract_dict:
    prediction_dict[key] = [data[str(index)]["prediction"] == data[str(index)]["gold"] for index in abstract_dict[key]]

hard_count, soft_count = 0, 0
fact_count = 0
for key in prediction_dict:
    if len(prediction_dict[key]) < 2:
        continue
    fact_count += 1
    if all(prediction_dict[key]):
        hard_count += 1
    soft_count += prediction_dict[key].count(True) / len(prediction_dict[key])

print(f"[AbsAcc]: {hard_count / fact_count}")

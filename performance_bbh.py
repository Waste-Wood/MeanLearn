import json
import os
from argparse import ArgumentParser
import re


def bbh_freeform_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    ans = ans.split('\n')[0]
    if ans.endswith('.'):
        ans = ans[:-1]
    return ans


def process_classification(f):
    count, size = 0, 0
    base_index = ord("A")
    data = json.load(open(os.path.join(opt.input_dir, "classification", f), "r"))
    for index in data:
        losses = [data[index][d]['Loss'] for d in data[index] if "label" in d]
        label = re.findall(pattern, data[index]['gold'])[0]
        prediction = chr(base_index + losses.index(min(losses)))
        data[index]['prediction'] = prediction
        if label == prediction:
            count += 1
        size += 1
    fo = open(os.path.join(opt.input_dir, "classification", f), "w")
    json.dump(data, fo, indent=4)
    return count, size


def process_generation(f):
    data = json.load(open(os.path.join(opt.input_dir, f), "r"))
    references = [data[index]['gold'] for index in data]
    predictions = [data[index]['prediction'] for index in data]
    if len(predictions) != len(references):
        return {
            'error': 'predictions and references have different '
            'length'
        }
    predictions = [bbh_freeform_postprocess(pred) for pred in predictions]

    cnt = 0
    for pred, ref in zip(predictions, references):
        if pred == ref:
            cnt += 1

    return cnt, len(predictions)


parser = ArgumentParser()
parser.add_argument("--input_dir", type=str, default="")
opt = parser.parse_args()

classification_files = [f for f in os.listdir(os.path.join(opt.input_dir, "classification")) if f.endswith(".json")]
generation_files = [f for f in os.listdir(opt.input_dir) if f not in classification_files and f.endswith(".json")]

pattern = re.compile("[A-Z]")

all_count, all_size = 0, 0
classification_count, classification_size = 0, 0
for f in classification_files:
    count, size = process_classification(f)
    all_count += count
    all_size += size
    # print(f"[{f}]:\t{count / size}")
classification_count, classification_size = all_count, all_size

generation_count, generation_size = 0, 0
for d in generation_files:
    count, size = process_generation(d)
    all_count += count
    all_size += size
generation_count, generation_size = all_count - classification_count, all_size - classification_size

print(f"[TOTAL]:\t{all_count / all_size}")







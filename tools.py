import jsonlines
from torch.utils.data import Dataset
from transformers import LlamaTokenizer
import torch


def read_jsonl(path):
    premises, hypotheses1, hypotheses2, rules, labels = [], [], [], [], []
    with jsonlines.open(path) as reader:
        for line in reader:
            premises.append(line["premise"])
            hypotheses1.append(line["hypothesis1"])
            hypotheses2.append(line["hypothesis2"])
            rules.append(line["conceptual_explanation"])
            labels.append(line["label"])
    return premises, hypotheses1, hypotheses2, rules, labels


def read_test_jsonl(path):
    inputs, outputs, systems = [], [], []
    fi = jsonlines.open(path, "r")
    for line in fi:
        inputs.append(line["input"])
        try:
            outputs.append(line["output"])
        except:
            outputs.append(line["target"])
        if "system" in line:
            systems.append(line["system"])
    fi.close()
    if len(systems) == 0:
        return inputs, outputs
    else:
        return inputs, outputs, systems


class DynamicDataset(Dataset):
    def __init__(self, *args):
        super(DynamicDataset).__init__()
        self.args = args
    
    def __len__(self):
        return len(self.args[0])

    def __getitem__(self, index):
        return tuple(arg[index] for arg in self.args)


def collate_fn(data):
    return tuple(d[0] for d in data)


def process_dialogs(dialogs, tokenizer: LlamaTokenizer):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
    UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."
    prompt_tokens = []
    for dialog in dialogs:
        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        dialog_tokens = sum(
            [
                tokenizer.encode(
                    f"<s>{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} </s>",
                    # bos=True,
                    # eos=True,
                )
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"<s>{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
            # bos=True,
            # eos=False,
        )
        prompt_tokens.append(dialog_tokens)
    
    bsz = len(prompt_tokens)
    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    total_len = 10 + max_prompt_len
    pad_id = tokenizer.pad_token_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    input_text_mask = tokens != pad_id

    return tokens, input_text_mask


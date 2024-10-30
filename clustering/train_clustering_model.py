from transformers import RobertaModel, RobertaTokenizer
import jsonlines
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch


data = [d for d in jsonlines.open("./data/train.jsonl", "r")]

model = RobertaModel.from_pretrained("/home/kxiong/huggingface_transformers/roberta-large")
tokenizer = RobertaTokenizer.from_pretrained("/home/kxiong/huggingface_transformers/roberta-large")
model = model.cuda()

inputs, labels = [], []
for d in data:
    inputs += [d["example1"], d["example2"]]
    labels += [d["label"], -1]
tokenizer_outputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", max_length=512)
DATA = TensorDataset(tokenizer_outputs["input_ids"], tokenizer_outputs["attention_mask"], torch.LongTensor(labels))
loader = DataLoader(DATA, batch_size=256, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train()

epochs = 5
loss_func = torch.nn.CosineEmbeddingLoss()

for epoch in range(epochs):
    total_loss = 0
    print(f"Epoch: {epoch}")
    for batch in tqdm(loader):
        optimizer.zero_grad()
        batch = tuple(t.cuda() for t in batch)
        input_ids, attention_mask, label = batch
        outputs = model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        loss = loss_func(outputs[::2], outputs[1::2], label[::2])

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss/len(loader)}")

model.save_pretrained("./model")





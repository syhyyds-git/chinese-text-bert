import torch
from torch.utils.data import DataLoader
from model import BertEncoder, BertSimFinetune
from finetune_dataset import LCQMCDataset
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)

train_dataset=LCQMCDataset("train.tsv",vocab)
dev_dataset=LCQMCDataset("dev.tsv",vocab)
test_dataset=LCQMCDataset("test.tsv",vocab)

train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
dev_loader=DataLoader(dev_dataset,batch_size=32)
test_loader=DataLoader(test_dataset,batch_size=32)


encoder = BertEncoder(len(vocab))

state_dict = torch.load('bert_pretrained.pt', map_location=device)

encoder_state_dict = {k.replace("encoder.", ""): v for k,v in state_dict.items() if k.startswith("encoder.")}
missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=False)
print("=== 权重对比 ===")
print("缺失参数:", missing)
print("多余参数:", unexpected)
encoder.to(device)

model = BertSimFinetune(encoder).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
best_f1 = 0

for epoch in range(30):
     total_loss = 0
     model.train()
     for input_ids, seg_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        seg_ids = seg_ids.to(device)
        labels = labels.to(device)
        attention_mask = (input_ids != vocab['[PAD]']).long().to(device)

        logits=model(input_ids,seg_ids,attention_mask)
        loss=torch.nn.functional.cross_entropy(logits,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
     avg_loss = total_loss / len(train_loader)
     print(f"[Finetune] Epoch {epoch+1} train_loss: {avg_loss:.4f}")   


model.eval()
with torch.no_grad:

    all_preds,all_labels=[],[]
    for input_ids, seg_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        seg_ids = seg_ids.to(device)
        labels = labels.to(device)
        attention_mask = (input_ids != vocab['[PAD]']).long().to(device)

        logits=model(input_ids,seg_ids,attention_mask)
        preds=torch.argmax(logits,dim=-1)

        all_preds.extend(preds.cpu().tolist)
        all_labels.extend(labels.cpu().tolist())
    pre = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Dev | P={pre:.4f} R={rec:.4f} F1={f1:.4f}")

    if f1>best_f1:
        best_f1=f1
        torch.save(model.state_dict(),'bert_lcqc_best.pt')
        

print("=== 测试集评估 ===")
model.load_state_dict(torch.load('bert_lcqc_best.pt'))
model.eval()
with torch.no_grad:
    for input_ids, seg_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        seg_ids = seg_ids.to(device)
        labels = labels.to(device)
        attention_mask = (input_ids != vocab['[PAD]']).long().to(device)

        logits=model(input_ids,seg_ids,attention_mask)
        preds=torch.argmax(logits,dim=-1)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        
pre = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
print(f"Test | P={pre:.4f} R={rec:.4f} F1={f1:.4f}")
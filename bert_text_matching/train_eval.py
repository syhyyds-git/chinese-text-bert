# train_eval.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attn = batch['attention_mask'].to(device)
        token_type = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids, attn, token_type)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            token_type = batch['token_type_ids'].to(device)
            label = batch['label'].to(device)

            logits = model(input_ids, attn, token_type)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().tolist())
            labels.extend(label.cpu().tolist())

    acc = accuracy_score(labels, preds)
    pre = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return acc, pre, rec, f1

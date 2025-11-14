# main.py
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW
from model import BertSimModel
from dataset import LCQMCDataset
from train_eval import train, evaluate
from utils import save_model, load_model

def main():
    pretrained = 'bert-base-chinese'  # 可改 'hfl/chinese-macbert-base'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    """
     #输入文本，可分词，加入[CLS]和[SEP]，
     #转换为ID序列，填充或截断到max_len，
     # 返回ID序列和注意力掩码和token_type_ids
     token_type_ids:[[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("加载数据中...")
    train_dataset = LCQMCDataset('train.tsv', tokenizer)
    dev_dataset = LCQMCDataset('dev.tsv', tokenizer)
    test_dataset = LCQMCDataset('test.tsv', tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print("构建模型中...")
    model = BertSimModel(pretrained).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    best_f1 = 0
    for epoch in range(3):
        print(f"\n===== Epoch {epoch + 1} =====")
        loss = train(model, train_loader, optimizer, device)
        acc, pre, rec, f1 = evaluate(model, dev_loader, device)
        print(f"Dev | Loss={loss:.4f} | Acc={acc:.4f} | P={pre:.4f} | R={rec:.4f} | F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            save_model(model, "best_model.pt")

    print("\n加载最优模型并在测试集上评估...")
    model = load_model(model, "best_model.pt", device)
    acc, pre, rec, f1 = evaluate(model, dev_loader, device)
    print(f"Test | Acc={acc:.4f} | P={pre:.4f} | R={rec:.4f} | F1={f1:.4f}")

if __name__ == "__main__":
    main()

# dataset.py
import torch
from torch.utils.data import Dataset

class LCQMCDataset(Dataset):
    """
    LCQMC 数据集加载器
    """
    def __init__(self, path, tokenizer, max_len=64):
        self.samples = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s1, s2, label = line.split('\t')
                    self.samples.append((s1, s2, int(label)))
                except:
                    continue
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s1, s2, label = self.samples[idx]
        encoded = self.tokenizer(
            s1, s2,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'#返回 PyTorch 张量
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
             #原张量 shape 是 [1, seq_len],squeeze(0） 后[seq_len]
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'token_type_ids': encoded['token_type_ids'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

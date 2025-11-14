import torch
from torch.utils.data import Dataset
import pandas as pd
class LCQMCDataset(Dataset):
    def __init__(self,file_path,vocab,max_len=64):
        self.voca=vocab
        self.max_len=max_len
        self.df=pd.read_csv(file_path,sep="\t")
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        sent1,sent2,label=self.df.iloc[idx]
        tokens=['[CLS]']+list(str(sent1)) +['[SEP]']+list(str(sent2))+['[SEP]']
        seg_ids=[0]*(len(sent1)+2)+[1]*(len(sent2)+1)

        input_ids = [self.vocab.get(t, self.vocab['[UNK]']) for t in tokens]

 
        if len(input_ids) < self.max_len:
            pad_len = self.max_len - len(input_ids)
            input_ids += [self.vocab['[PAD]']]*pad_len
            seg_ids += [0]*pad_len
        else:
            input_ids = input_ids[:self.max_len]
            seg_ids = seg_ids[:self.max_len]
        return (torch.tensor(input_ids),torch.tensor(seg_ids),torch.tensor(label))
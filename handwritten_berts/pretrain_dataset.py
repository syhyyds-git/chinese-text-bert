import torch
from torch.utils.data import Dataset
import random
class BertPretrainDataset(Dataset):
    def __init__(self,corpus_file,vocab,max_len=64,mlm_prob=0.15):
        self.max_len=max_len
        self.mlm_prob=mlm_prob
        self.vocab=vocab
        self.vocab_size = len(self.vocab)
        with open(corpus_file,'r',encoding='utf-8') as f:
            self.sentences=[line.strip() for line in f if line.strip()]
    def __len__(self):
        return len(self.sentences)-1
    def __getitem__(self, idx):
        sent_a=self.sentences[idx]

        if random.random()<0.5:
            sent_b=self.sentences[idx+1]
            nsp_label=1
        else:
            sent_b=random.choice(self.sentences)
            nsp_label=0

        tokens=['[CLS]']+list(sent_a) +['[SEP]'] +list(sent_b)+['[SEP]']

        input_ids=[self.vocab.get(token,self.vocab['[UNK]']) for token in tokens]
        seg_ids=[0]*(2+len(sent_a))+[1]*(1+len(sent_b))

        pad_len=self.max_len-len(input_ids)
        if pad_len>0:
            attn_ids=[1]*len(input_ids)+[0]*pad_len
            input_ids+=[self.vocab['[PAD]']]*pad_len
            seg_ids+=[0]*pad_len
        else:
            attn_ids=[1]*self.max_len
            input_ids=input_ids[:self.max_len]
            seg_ids=seg_ids[:self.max_len]
        mlm_labels=[-100]*self.max_len

        for i in range(self.max_len):
            if i>=len(tokens) or  tokens[i] in['[CLS]','[SEP]']:
                continue
            if random.random()<self.mlm_prob:
                mlm_labels[i]=input_ids[i]
                prob = random.random()
                if prob<0.8:
                    input_ids[i]=self.vocab['[MASK]']
                elif prob<0.9:
                    input_ids[i]=random.randint(0, self.vocab_size - 1)
        return (torch.tensor(input_ids),
                torch.tensor(seg_ids),
                torch.tensor(attn_ids),
                torch.tensor(mlm_labels),
                torch.tensor(nsp_label)
               )









    
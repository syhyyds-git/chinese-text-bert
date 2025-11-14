class SimpleTokenizer:
    def __init__(self,vocab):
        self.vocab=vocab
        self.idx2token={i:t for t,i in self.vocab.items()}
        self.vocab_size=len(self.vocab)
    def tokenize(self,text):
        return list(text.strip())
    def encode(self,tokens,max_len):
        input_ids=[self.vocab.get(t,self.vocab['[UNK]']) for t in tokens]
        input_ids=input_ids[:max_len]
        pad_len=max_len-len(input_ids)
        input_ids+=[self.vocab['[PAD]']]*pad_len
        return input_ids
    def encode_pair(self,token_a,token_b,max_len):
        tokens=['[CLS]']+token_a+['[SEP]']+token_b+['[SEP]']
        seg_ids=[0]*(len(token_a)+2)+[1]*(len(token_b)+1)
        input_ids=self.encode(tokens,max_len)
        seg_ids=seg_ids[:max_len]+[0]*(max_len-len(seg_ids))
        atten_mask=[1 if id!=self.vocab['[PAD]'] else 0 for id in input_ids]
        return input_ids,seg_ids,atten_mask

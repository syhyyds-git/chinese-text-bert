import torch
import json
from model import BertEncoder, BertSimFinetune
from tokenizer import SimpleTokenizer
class SimplePredictor:
    def __init__(self,model_path="bert_lcqc_best.pt",vocab_path="vocab.json",max_len=64):
        self.device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer=SimpleTokenizer(self.vocab)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.encoder=BertEncoder(len(self.vocab),max_len)
        self.model=BertSimFinetune(self.encoder)
        self.model.load_state_dict(torch.load(model_path,map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.max_len=max_len
    def predict(self,text_a,text_b):
        tokens1=list(text_a.strip())
        tokens2=list(text_b.strip())
        input_ids,seg_ids,attn_mask=self.tokenizer.encode_pair(tokens1,tokens2,self.max_len)
        input_ids=torch.tensor(input_ids,device=self.device).unsqueeze(0)
        seg_ids=torch.tensor(seg_ids,device=self.device).unsqueeze(0)
        attn_mask=torch.tensor(attn_mask,device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits=self.model(input_ids,seg_ids,attn_mask)
            probs=torch.softmax(logits,dim=1)
            pred=torch.argmax(probs,dim=1).item()
        return pred
if __name__ == "__main__":
    predictor = SimplePredictor(model_path="bert_lcqc_best.pt", vocab_path="vocab.json")
    s1 = "书桌在衣柜的东边？"
    s2 = "衣柜在书桌的西边？"
  
    result = predictor.predict(s1, s2)
    print(f"句子1: {s1}")
    print(f"句子2: {s2}")
    print(f"是否相似: {result}")

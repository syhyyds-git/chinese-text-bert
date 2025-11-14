import torch
import torch.nn as nn
import math
class MultiHeadSelfAttention(nn.Module):
     def __init__(self,d_model=128,num_heads=4):
        super().__init__()
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.q=nn.Linear(d_model,self.d_k*num_heads)
        self.k=nn.Linear(d_model,self.d_k*num_heads)
        self.v=nn.Linear(d_model,self.d_k*num_heads)
        self.out=nn.Linear(d_model,d_model)
     def forward(self,x,attn_mask):#[b,t]
         B,T,D=x.size()
         Q=self.q(x).view(B,T,self.num_heads,self.d_k).transpose(1,2)
         K=self.k(x).view(B,T,self.num_heads,self.d_k).transpose(1,2)
         V=self.v(x).view(B,T,self.num_heads,self.d_k).transpose(1,2)
         attn_scores=Q@K.transpose(-2,-1)/math.sqrt(self.d_k)#[b,num_heads,t,t]
         if attn_mask is not None:
             attn_mask=attn_mask.unsqueeze(1).unsqueeze(2)
             attn_scores=attn_scores.masked_fill(attn_mask==0,-1e9)
         attn_scores=torch.softmax(attn_scores,dim=-1)
         attn_output=attn_scores@V#[b,num_heads,t,d_k]
         attn_output=attn_output.transpose(1,2).contiguous().view(B,T,D)
         """
         因为 transpose() 只是改变视图，不改变内存排列，
         所以现在这个张量是 非连续的 (non-contiguous)。
         contiguous()让张量真正复制成连续内存
         """
         return self.out(attn_output)
             
             

class TransformerBlock(nn.Module):
    def __init__(self,d_model=128,d_ff=512,num_heads=4):
        super().__init__()
        self.attn=MultiHeadSelfAttention(d_model,num_heads)
        self.norm1=nn.LayerNorm(d_model)
        self.ff=nn.Sequential(nn.Linear(d_model,d_ff),nn.ReLU(),nn.Linear(d_ff,d_model))
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(0.1)
    def forward(self,x,attn_mask):
        attn_output=self.attn(x,attn_mask)
        x=self.norm1(x+self.dropout(attn_output))
        ff_output=self.ff(x)
        ff_output=self.norm2(x+self.dropout(ff_output))
        return x
class BertEncoder(nn.Module):
    
     def __init__(self,vocab_size,max_len=64,d_model=128,num_layers=6,d_ff=512,num_heads=4):
         super().__init__()
         self.token_embed=nn.Embedding(vocab_size,d_model)
         self.pos_embed=nn.Embedding(max_len,d_model)
         self.seg_embed=nn.Embedding(2,d_model)
         self.layers=nn.ModuleList(TransformerBlock(d_model,d_ff,num_heads) for _ in range(num_layers))
         self.norm=nn.LayerNorm(d_model)
         self.droput=nn.Dropout(0.1)   
     def forward(self,input_ids,seg_ids,attn_mask):
         b,seq_len=input_ids.size()
         token_embed=self.token_embed(input_ids)
         seg_embed=self.seg_embed(seg_ids)
         pos=torch.arange(seq_len,device=input_ids.device).unsqueeze(0).expand(b,-1)
         pos_embed=self.pos_embed(pos)
         x=token_embed+seg_embed+pos_embed
         x=self.droput(x)

         for layer in self.layers:
             x= layer(x,attn_mask)
         return self.norm(x)
           
class BertPretrain(nn.Module):
    def __init__(self,vocab_size,d_model=128):
        super().__init__()
        self.encoder=BertEncoder(vocab_size)
        self.mlm_head=nn.Linear(d_model,vocab_size)
        self.nsp_head=nn.Linear(d_model,2)
    def forward(self,input_ids,seg_ids,attn_mask):
        output=self.encoder(input_ids,seg_ids,attn_mask)
        mlm_output=self.mlm_head(output)
        nsp_outhead=self.nsp_head(output[:,0])
        return mlm_output,nsp_outhead
class BertSimFinetune(nn.Module):
     def __init__(self,bert_encoder,num_classes=2):
        super().__init__()
        self.encoder=bert_encoder
        self.classifier=nn.Linear(bert_encoder.d_model,num_classes)
        self.dropout=nn.Dropout(0.1)
     def forward(self,input_ids,seg_ids,attn_mask):
       h=self.encoder(input_ids,seg_ids,attn_mask)
       cls_vec=h[:,0,:]
       logits=self.classifier(self.dropout(cls_vec))
       return logits

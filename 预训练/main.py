import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
def  make_batch():
    batch=[]
    positive=negative=0
    while positive!=batch_size/2 or negative!=batch_size/2:
        tokens_a_index,tokens_b_index=randrange(len(sentences)),randrange(len(sentences))
        if tokens_a_index+1==tokens_b_index and positive==batch_size/2:
            continue
        elif tokens_a_index+1!=tokens_b_index and negative==batch_size/2:
            continue

        tokens_a,tokens_b=token_list[tokens_a_index],token_list[tokens_b_index]
        input_ids=[word_dict['[CLS]']]+tokens_a+[word_dict['[SEP]']]+tokens_b+[word_dict['[SEP]']]

        segment_ids=[0]*(1+len(tokens_a)+1)+[1]*(len(tokens_b)+1)
        n_pred=min(max_pred,max(1,int(round(len(input_ids)*0.15))))
        
        cand_maked_pos=[i for i,token in enumerate(input_ids)
                        if token!=word_dict['[CLS]'] and token!=word_dict['[SEP]'] ]
        shuffle(cand_maked_pos)

        masked_tokens,masked_pos=[],[]
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random()<0.8:
                input_ids[pos]=word_dict['[MASK]']
            elif random()<0.5:
                input_ids[pos]=randint(0,vocab_size-1)

        n_pad=maxlen-len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        # 在segment_ids 后面补零；这里有一个问题，0和之前的重了，这里主要是为了区分不同的句子，所以无所谓啊；他其实是另一种维度的位置信息；

        if max_pred>n_pred:
            n_pad=max_pred-n_pred
            masked_tokens.extend([0]*n_pad)
            masked_pos.extend([0]*n_pad)
        
        if tokens_a_index+1==tokens_b_index and positive<batch_size/2:
            batch.append([input_ids,segment_ids,masked_tokens,masked_pos,True])
            positive+=1
        elif tokens_a_index+1!=tokens_b_index and negative<batch_size/2:
            batch.append([input_ids,segment_ids,masked_tokens,masked_pos,False])
            negative+=1
    return batch
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding,self).__init__()
        self.tok_embed=nn.Embedding(vocab_size,d_model)#token embedding
        self.pos_embed=nn.Embedding(maxlen,d_model)#position embedding
        self.seg_embed=nn.Embedding(n_segments,d_model)#segment(token type) embedding
        self.norm=nn.LayerNorm(d_model)
        #把每层的输出调整到分布稳定、均值为 0、方差为 1的状态，使训练更平稳
    def forward(self,x,seg):
        seq_len=x.size(1)
        pos=torch.arange(seq_len,dtype=torch.long)
        pos=pos.unsqueeze(0).expand_as(x) #(seq,)->(batch_size,seq_len)
        embedding=self.tok_embed(x)+self.pos_embed(pos)+self.seg_embed(seg)
        return self.norm(embedding)
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention,self).__init__()

    def forward(self,Q,K,V,attn_mask):
        scores=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(d_k)
        scores.masked_fill_(attn_mask,-1e9)
        #处理变长序列时的填充（Padding）掩码：为了将不同长度的序列批量处理
        # 我们会将序列填充到相同长度，然后通过掩码标记填充位置，
        # 使模型不关注这些填充位置,和bert MLM任务预测词无关。
        attn=nn.Softmax(dim=-1)(scores)
        context=torch.matmul(attn,V)
        return context,attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        self.W_Q=nn.Linear(d_model,d_k*n_head)
        self.W_K=nn.Linear(d_model,d_k*n_head)
        self.W_V=nn.Linear(d_model,d_v*n_head)
        self.fc=nn.Linear(n_head*d_v,d_model)
        self.layer_norm=nn.LayerNorm(d_model)
    def forward(self,Q,K,V,attn_mask):
       """
       q:[b,len_q,d_model]
       k:[b,len_k,d_model]
       v:[b,len_v,d_model]
       """
       residual,batch_size=Q,Q.size(0)
       q_s=self.W_Q(Q).view(batch_size,-1,n_head,d_k).transpose(1,2)#[b,n_head,len_q,len_k]
       k_s=self.W_K(K).view(batch_size,-1,n_head,d_k).transpose(1,2)#[b,n_head,len_k,len_k]
       v_s=self.W_V(V).view(batch_size,-1,n_head,d_v).transpose(1,2)#[b,n_head,len_v,len_v]

       attn_mask=attn_mask.unsqueeze(1).repeat(1,n_head,1,1)#[b,n_head,len_q,len_k]
       
       context,attn=ScaledDotProductAttention()(q_s,k_s,v_s,attn_mask)
       # context: [b,n_heads,len_q,d_v]
       # attn: [batch_size,n_heads,len_q(=len_k),len_k(=len_q)]
       context=context.transpose(1,2).contiguous().view(batch_size,-1,n_head*d_v)
       # context: [batch_size,len_q,n_heads * d_v]
       #contiguous() 方法确保张量在内存中是连续存储的。当张量不连续时，某些操作（特别是 view()）会失败。
       out=self.fc(context)
       return self.layer_norm(out+residual),attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet,self).__init__()
        self.fc1=nn.Linear(d_model,d_ff)
        self.fc2=nn.Linear(d_ff,d_model)
        self.gelu=nn.GELU()
    def forward(self,x):
        return self.fc2(self.gelu(self.fc1(x)))
        #GELU(x) = x * Φ(x)
        #GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer,self).__init__()
        self.enc_self_attn=MultiHeadAttention()
        self.pos_ffn=PoswiseFeedForwardNet()
    def forward(self,enc_inputs,enc_self_attn_mask):
        enc_outputs,attn=self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)# enc_inputs to same Q,K,V
        enc_outputs=self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs,attn
def get_attn_pad_mask(seq_q,seq_k):
    batch_size,len_q=seq_q.size()
    batch_size,len_k=seq_k.size()
    pad_attn_mask=seq_k.data.eq(0).unsqueeze(1)#[b,1,len_k]
    #eq()是一个张量的方法，用于逐元素比较是否等于给定值。
    #它返回一个布尔张量（值为0或1），其中1表示对应位置等于给定值，
    #0则表示不等于。
    return pad_attn_mask.expand(batch_size,len_q,len_k)#[b,len_q,len_k]
class BERT(nn.Module):
    def __init__(self):
        super(BERT,self).__init__()
        self.embedding=Embedding()#词向量层
        self.layers=nn.ModuleList([EncoderLayer() for _ in range(n_layers)])#把N个encoder堆叠起来

        self.fc=nn.Linear(d_model,d_model)
        self.activ1=nn.Tanh()
        self.linear=nn.Linear(d_model,d_model)
        self.norm=nn.LayerNorm(d_model)
        self.classifier=nn.Linear(d_model,2)
        # MLM解码部分，从Embedding_size解码到词表大小
        embed_weight=self.embedding.tok_embed.weight
        
        self.activ2=nn.GELU()
        n_vocab,n_dim=embed_weight.size()
        self.decoder=nn.Linear(n_dim,n_vocab,bias=False)
        self.decoder.weight=embed_weight
        #[vocab_size, d_model],实现了权重共享，即输入嵌入矩阵和解码输出层共享同一个权重矩阵
        #大幅减少参数
        #增强表示一致性：输入和输出共享相同的语义空间
        #在计算时会自动转置，所以这里不需要手动转置
        self.decoder.bias=nn.Parameter(torch.zeros(n_vocab))
    def forward(self,input_ids,segment_ids,masked_pos):

        output=self.embedding(input_ids,segment_ids)
        enc_self_attn_mask=get_attn_pad_mask(input_ids,input_ids)
        #补句长的pad，用于注意力计算，避免关注填充位置
        for layer in self.layers:
            output,enc_self_attn=layer(output,enc_self_attn_mask)
        # output : [batch_size, len, d_model]
        # attn : [batch_size, n_heads, d_mode, d_model]
        h_pooled=self.activ1(self.fc((output[:,0])))#[b,d_model]
        logits_clsf=self.classifier(h_pooled)#[b,2]

        masked_pos=masked_pos[:,:,None].expand(-1,-1,output.size(-1))
        #[b,max_pred,d_model],为了收集output中对应masked_pos的向量
        h_masked=torch.gather(output,1,masked_pos)
        #收集output在维度1上的对应masked_pos的向量
        h_masked=self.norm(self.activ2(self.linear(h_masked)))
        logits_lm=self.decoder(h_masked)+self.decoder.bias
        # [batch_size, max_pred, n_vocab]  解码到词表大小
        return logits_lm,logits_clsf
    





            
        





if __name__ =='__main__':
    maxlen=30
    batch_size=32
    max_pred=5
    n_layers=6
    n_head=12
    d_model=768
    d_ff=3072
    d_k=d_v=64
    n_segments=2

    text=(
        'Hello, how are you? I am Romeo.\n'
        'Hello, Romeo My name is Juliet. Nice to meet you.\n'
        'Nice meet you too. How are you today?\n'
        'Great. My baseball team won the competition.\n'
        'Oh Congratulations, Juliet\n'
        'Thanks you Romeo'
    )
    sentences=re.sub("[.,!?\\-]",'',text.lower()).split('\n')
    #print(sentences)
    word_list=list(set(" ".join(sentences).split()))
    word_dict={'[PAD]':0,'[CLS]':1,'[SEP]':2,'[MASK]':3}
    for i,word in enumerate(word_list):
        word_dict[word]=i+4
    num_dict={i:w for w,i in word_dict.items()}
    vocab_size = len(word_dict)

    token_list=[]
    for sentence in sentences:
        arr=[word_dict[word] for word in sentence.split()]
        token_list.append(arr)
    
    model=BERT()
    criterion=nn.CrossEntropyLoss(ignore_index=0)
    optimizer=optim.Adam(model.parameters(),lr=1e-3)

    for epoch in range(100):

        optimizer.zero_grad()
        batch=make_batch()
        input_ids,segment_ids,masked_tokens,masked_pos,isNext=map(torch.LongTensor,zip(*batch))
        logits_lm,logits_clsf=model(input_ids,segment_ids,masked_pos)
        
        loss_lm=criterion(logits_lm.transpose(1,2),masked_tokens)
        #CrossEntropyLoss期望vocab_size在第1维
        loss_lm=(loss_lm.float()).mean()
        loss_clsf=criterion(logits_clsf,isNext)
        loss=loss_lm+loss_clsf
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss)) 


    model.eval()
    with torch.no_grad():
        input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[0]  # 第一个样本
        print(text)
        print('================================')
        print([num_dict[w] for w in input_ids if num_dict[w] != '[PAD]'])

        logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), \
                            torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
        logits_lm = logits_lm.data.argmax(2)[0].data.numpy()

          # 使用masked_pos来确定哪些是真实的掩码位置
        
        real_mask_indices = [i for i, pos in enumerate(masked_pos) if pos != 0]
    
        real_masked_tokens = [masked_tokens[i] for i in real_mask_indices]
        predicted_masked_tokens = [logits_lm[i] for i in real_mask_indices]
    
        print('真实掩码token ID:', real_masked_tokens)
        print('预测掩码token ID:', predicted_masked_tokens)


       

        logits_clsf = logits_clsf.data.argmax(1)[0].data.numpy()
        print('isNext : ', True if isNext else False)
        print('predict isNext : ',True if logits_clsf else False)





    


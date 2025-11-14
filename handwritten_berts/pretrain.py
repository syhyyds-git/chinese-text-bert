import torch
from torch.utils.data import DataLoader
from model import BertPretrain
from pretrain_dataset import BertPretrainDataset
import sentencepiece as spm
import json

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('vocab.json','r',encoding='utf-8') as f:
    vocab=json.load(f)


dataset=BertPretrainDataset('corpus.txt',vocab)
loader=DataLoader(dataset,batch_size=16,shuffle=True)


model=BertPretrain(len(vocab),d_model=128).to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)


for epoch in range(30):
    total_loss=0
    minn_loss=0
    model.train()
    for input_ids,seg_ids,attn_mask,mlm_labels,nsp_labels in loader:
        input_ids = input_ids.to(device)
        seg_ids = seg_ids.to(device)
        mlm_labels = mlm_labels.to(device)
        nsp_labels = nsp_labels.to(device)
        attn_mask=attn_mask.to(device)
        mlm_logits,nsp_logits=model(input_ids,seg_ids,attn_mask)
        loss_mlm=torch.nn.functional.cross_entropy(mlm_logits.view(-1,mlm_logits.size(-1)),
                                                   mlm_labels.view(-1),ignore_index=-100)
        loss_nsp=torch.nn.functional.cross_entropy(nsp_logits.view(-1,nsp_logits.size(-1)),
                                                   nsp_labels.view(-1))
        loss=loss_mlm+loss_nsp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(loader)}')
    if total_loss<minn_loss:
        torch.save(model.state_dict(), 'bert_pretrained.pt')
        print("save model")
print("预训练完成")












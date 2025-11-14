# model.py
import torch
import torch.nn as nn
from transformers import BertModel

class BertSimModel(nn.Module):
    """
    BERT文本匹配模型：使用[CLS]向量做二分类
    """
    def __init__(self, pretrained_model='bert-base-chinese', dropout=0.3):
        super(BertSimModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids#seg_id
        )
        pooled_output = outputs.pooler_output  # [CLS]
        out = self.dropout(pooled_output)
        logits = self.classifier(out)
        return logits

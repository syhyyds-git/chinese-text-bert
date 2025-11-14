# chinese-text-bert
## 基于BERT的文本相似度判断。

## 设备要求:

我自己用的4GB显存的笔记本电脑，handwritten_berts这个代码预训练大概跑了4-5小时，微调3-4小时。

## 文件管理

### 1.预训练

用一个小例子手写一下预训练过程。

具体参考：https://zhuanlan.zhihu.com/p/605020970

### 2.bert_text_matching

调用已有的，训练好的bert-base-chinese模型进行文本相似度判断

###3.handwritten_berts
####手写预训练，微调过程
####3.1generate_corpus_nsp_mlm.py
产生预训练文本，可调生成文本行数，我用的是50万行
####3.2build——vocab.py
分词，产生词表
####3.3pretrain.py
预训练
####3.4finetune.py
微调过程
####3.5predict.py
输入两句话，可以判断两句话是否相似

###4.训练结果：
####4.1.pretrain.py
#####Epoch 28, Loss: 3.490206859298706
#####save model
#####Epoch 29, Loss: 3.482210868400574
#####save model
#####Epoch 30, Loss: 3.4755539898605345
#####save model
损失函数大概到3.5左右
####4.2.finetune.py
#####[Finetune] Epoch 28 train_loss: 0.2785
#####Dev | P=0.8993 R=0.9392 F1=0.9188
#####[Finetune] Epoch 29 train_loss: 0.2732
#####Dev | P=0.8684 R=0.9642 F1=0.9138
#####[Finetune] Epoch 30 train_loss: 0.2695
#####Dev | P=0.9040 R=0.9397 F1=0.9215
准确率大概89-90，f1大概能到91
###5.如果有更好的设备条件，修改以下参数可能训练效果会更好
###5.1可以修改一下pretain.py第18行：
model=BertPretrain(len(vocab),d_model=128).to(device)
d_model可以改成256或则512
###5.2可以修改一下model.py第50行
def __init__(self,vocab_size,max_len=64,d_model=128,num_layers=6,d_ff=512,num_heads=4):
d_ff可以修改成1024，或则2048
num_heads可以修改成8
###5.3generate_corpus_nsp_mlm.py设定的文本行数可以更大



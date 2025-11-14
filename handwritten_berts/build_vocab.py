import sentencepiece as spm
import json

input_file='corpus.txt'
vocab_size=8000
vocab_prefix='spm'

spm_train=f'--input={input_file} --model_prefix={vocab_prefix}  --vocab_size={vocab_size}\
            --character_coverage=0.995 --model_type=bpe --pad_id=0 --unk_id=1\
                --bos_id=-1 --eos_id=-1'
spm.SentencePieceTrainer.train(spm_train)



sp=spm.SentencePieceProcessor()
sp.load(f'{vocab_prefix}.model')

vocab={'[PAD]':0,'[MASK]':1,'[CLS]':2,'[SEP]':3,'[UNK]':4}

offset=len(vocab)

for i in range(sp.get_piece_size()):
    piece=sp.id_to_piece(i)
    if piece not in vocab:
        vocab[piece]=i+offset
with open("vocab.json",'w',encoding='utf-8') as f:
    json.dump(vocab,f,ensure_ascii=False,indent=2)

print(f"词表生成完成，共 {len(vocab)} 个 token")



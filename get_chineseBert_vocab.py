vocab = '/data02/gob/model_hub/chineseBert/vocab.txt'

with open(vocab,'r') as fp:
    words = fp.read().strip().split('\n')
    words = [i for i in words if len(i) == 1]
with open('./data/vocab.txt','w',encoding='utf-8') as fp:
    for word in words:
        fp.write(word + '\n')

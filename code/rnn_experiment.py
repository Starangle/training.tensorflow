import tensorflow as tf

def read_data():
    words=open('data/PTB/ptb.train.txt',"r").read().replace("\n","<eos>").split()
    uwords=set(words)
    word2id=dict(zip(uwords,range(len(uwords))))

    train_words=words
    train_ids=[word2id[word] for word in train_words if word in word2id]

    test_words=open('data/PTB/ptb.test.txt',"r").read().replace("\n","<eos>").split()
    test_ids=[word2id[word] for word in test_words if word in word2id]

    valid_words=open('data/PTB/ptb.valid.txt',"r").read().replace("\n","<eos>").split()
    valid_ids=[word2id[word] for word in valid_words if word in word2id]

    return train_ids,test_ids,valid_ids

def rnn_model(features,labels,mode):
    pass
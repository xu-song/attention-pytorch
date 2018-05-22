# -*- coding: utf-8 -*-


######################################################################
# Field: label hypothesis premise
# label有三种 neutral contradiction entailment
######################################################################


from torchtext import data
from torchtext import datasets

from dataset.translation import *
from dataset.aspect import Label

use_pretrain = False
MAX_LENGTH = 10  # filter by MAX_LENGTH


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

    
def filterPairs(pairs, label=None):
    if label is None:
        return [pair for pair in pairs if filterPair(pair)]
    else:
        return [pair for pair in pairs if pair[2]==label and filterPair(pair)]



def read(reverse=False, share_lang=True):
    inputs = data.Field(lower=True)
    answers = data.Field(sequential=False)
    train_set, dev_set, test_set = datasets.SNLI.splits(inputs, answers)
    train_set = [[' '.join(pair.premise), ' '.join(pair.hypothesis), pair.label] 
                 for pair in train_set]
    test_set = [[' '.join(pair.premise), ' '.join(pair.hypothesis), pair.label] 
                for pair in test_set]
    
    if share_lang:
        lang = Lang('premise+hypothesis')
        return lang, lang, Label('label'), train_set, test_set
    else:  # 兼容不同词典
        return Lang('premise'), Lang('hypothesis'), Label('label'), train_set, test_set  


def prepareData(index=True, filter_label=None):
    input_lang, output_lang, label, train_set, test_set = read()
    print("Read train %s, test %s" % (len(train_set), len(test_set)))
    
    train_set = filterPairs(train_set, filter_label)
    test_set = filterPairs(test_set, filter_label)
    print("Trimmed to train %s, test %s" % (len(train_set), len(test_set)))

    allset = train_set + test_set
    if index:
        print("Counting words...")
        for data in allset:
            input_lang.addSentence(data[0])  
            output_lang.addSentence(data[1])
            label.addLabel(data[2])

        print("Counted words:")
        print(input_lang.name, 'vocab:', input_lang.n_words)
        print(output_lang.name, 'vocab:', output_lang.n_words)

    if use_pretrain:
        global pretrained_weight
        embed_path = '/root/xs/w2v/glove.6B.200d.txt'
        pretrained_weight = data_helpers.word2vec.load_embedding_vectors_glove(input_lang.word2index, embed_path, 200)
    if filter_label is None:
        return input_lang, output_lang, label, train_set, test_set
    else:
        return input_lang, output_lang, train_set, test_set


# 数据快照，包括基本的数据统计
def snapshot():
    inputs = data.Field(lower=True)
    answers = data.Field(sequential=False)
    train_set, dev_set, test_set = datasets.SNLI.splits(inputs, answers)
    print('train: %d, dev: %d, test: %d' % (len(train_set), len(dev_set), len(test_set)))
    
    for i in range(20):
        #print('[premise]: %s;[hypothesis]: %s;[label]: %s' % (' '.join(train_set[i].premise), ' '.join(train_set[i].hypothesis),train_set[i].label))
        print('| %s | %s | %s |' % (' '.join(train_set[i].premise), ' '.join(train_set[i].hypothesis),train_set[i].label))


if __name__ == '__main__':
    snapshot()
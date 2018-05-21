# -*- coding: utf-8 -*-


'''
全局变量
'''
data_dir = 'data/aspect/'
MAX_LENGTH = 20




'''
label数据，处理label与index的映射
'''
class Label:
    def __init__(self, name):
        self.name = name
        self.labels = []
        self.label2index = {}
        self.label2count = {}

    def addLabel(self, label):
        if label not in self.labels:
            self.label2index[label] = len(self.labels)
            self.labels.append(label)
            self.label2count[label] = 1
        else:
            self.label2count[label] += 1

            
import torch
def tensorFromLabel(label, label_name):
    index = label.label2index[label_name]
    return torch.tensor(index, dtype=torch.long, device=device).view(-1, 1)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# 读取aspect数据
def prepareData(index=True):
    input_lang, aspect_lang, label, trainset, testset = read()
    print("Read train %s, test %s" % (len(trainset), len(testset)))
    trainset = filterPairs(trainset)
    testset = filterPairs(testset)
    print("Trimmed to train %s, test %s" % (len(trainset), len(testset)))


    allset = trainset + testset
    if index:
        print("Counting words...")
        for data in allset:
            input_lang.addSentence(data[0])
            aspect_lang.addSentence(data[1])
            label.addLabel(data[2])

        print("Counted words:")
        print(input_lang.name, 'vocab:', input_lang.n_words)
        print(aspect_lang.name,':', aspect_lang.n_words, aspect_lang.word2count)
        print(label.name, len(label.labels), label.label2count)

    if use_pretrain:
        global pretrained_weight
        embed_path = '/root/xs/w2v/glove.6B.200d.txt'
        pretrained_weight = data_helpers.word2vec.load_embedding_vectors_glove(input_lang.word2index, embed_path, 200)
    return input_lang, aspect_lang, label, trainset, testset


   



def read():
    # download
    urls = ['http://**.zip']
    dirname = 'snli_1.0'
    name = 'snli'
    
    print("Reading aspect...")
    lines1 = open('data/aspect/train.cor', encoding='utf-8').readlines()
    lines2 = open('data/aspect/test.cor', encoding='utf-8').readlines()
    
    train_set = [[lines1[i].strip(), lines1[i+1].strip(), lines1[i+2].strip()] for i in range(0, len(lines1),3)]
    test_set = [[lines2[i].strip(), lines2[i+1].strip(), lines2[i+2].strip()] for i in range(0, len(lines2),3)]
    return Lang('sent', False), Lang('aspect', False), Label('label'), train_set, test_set

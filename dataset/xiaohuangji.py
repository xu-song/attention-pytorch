# -*- coding: utf-8 -*-

from dataset.translation import Lang

use_pretrain = False
MAX_LENGTH = 10  # filter by MAX_LENGTH

urls = 'https://github.com/candlewill/Dialog_Corpus/raw/master/xiaohuangji50w_fenciA.conv.zip'


import config
args = config.get_args()

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

    
def filterPairs(pairs, label=None):
    if label is None:
        return [pair for pair in pairs if filterPair(pair)]
    else:
        return [pair for pair in pairs if pair[2]==label and filterPair(pair)]



def readXiaohuangji(share_lang=True):
    print("Reading lines...")
    lines = open(args.data_path + '/chat/xiaohuangji50w_fenciA.conv', encoding='utf-8').\
        read().strip().split('\nE')
    pairs = [[s.replace('/',' ').strip() for s in l.split('\nM')[1:]] for l in lines]
    if share_lang:
        lang = Lang('QA')
        return lang, lang, pairs
    else: # 兼容不同词典
        return Lang('Q'), Lang('A'), pairs



def prepareData(index=True):
    input_lang, output_lang, pairs = readXiaohuangji()    
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    if index:
        print("Counting words...")
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs




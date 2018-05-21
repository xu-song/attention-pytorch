# -*- coding: utf-8 -*-



def readXiaohuangji():
    print("Reading lines...")
    lines = open('data/chat/xiaohuangji50w_fenciA.conv', encoding='utf-8').\
        read().strip().split('\nE')

    pairs = [[s.replace('/',' ').strip() for s in l.split('\nM')[1:]] for l in lines]
    return Lang('Q'), Lang('A'), pairs


# 复用translation的代码
def prepareData(lang1, lang2, reverse=False, index=True):
    if dataset=='xiaohuangji':
        input_lang, output_lang, pairs = readXiaohuangji()
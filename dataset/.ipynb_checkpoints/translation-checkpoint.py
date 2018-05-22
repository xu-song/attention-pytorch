# -*- coding: utf-8 -*-

'''
原则上，翻译，对话，等多种数据都可用相同的数据结构，复用类。
'''

import unicodedata
import re
from io import open
import string


dataset = 'lang'
data_dir = 'data/translation/'

MAX_LENGTH = 10  # filter by MAX_LENGTH
SOS_token = 0
EOS_token = 1

# 英法翻译
urls = 'https://download.pytorch.org/tutorial/data.zip'

# 中英翻译


# 

'''
language
'''
class Lang:
    SOS_token = 0
    EOS_token = 1
    def __init__(self, name, count_sos=True):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.count_sos = count_sos
        if self.count_sos:
            self.index2word = {0: "SOS", 1: "EOS"}
            self.n_words = 2  # Count SOS and EOS
        else:
            self.index2word = {}
            self.n_words = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


    # Filtering vocabularies
    # trim by min_count, top_
    # 最好的方法是用treemap 来维护。


    # to string
    def snapshot():
        pass









######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase, and trim most
# punctuation.
#

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def normalizeZh(s):
    # s = re.sub(r"([)", r" \1", s)
    return None


######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs. reverse用于英法-法英翻译的互换。
#

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(data_dir + '%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs





######################################################################
# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).
#

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#


def prepareData(lang1, lang2, reverse=False, index=True):
    if dataset=='xiaohuangji':
        input_lang, output_lang, pairs = readXiaohuangji()
    elif dataset=='lang':
        input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    else:
        return

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



def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

'''
def indexesFromSentence(lang, sentence):
    sent = [lang.word2index[word] for word in sentence.split(' ')]
    global MAX_LENGTH
    #if (not is_crop) and len(sent)>MAX_LENGTH: 
    if len(sent)>MAX_LENGTH:
        MAX_LENGTH = len(sent)   # 如果不过滤掉超长length，那么就预设MAX_LENGTH=0，然后动态增加MAX_LENGTH
        print('MAX_LENGTH update:', MAX_LENGTH)
    return sent
'''



######################################################################
# convert to torch tensor
# ------------------

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def tensorFromSentence(lang, sentence, is_append_EOS = True):
    indexes = indexesFromSentence(lang, sentence)
    if is_append_EOS:
        indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)




if __name__ == '__main__':
    snapshot()

# Requirements

- pytorch >= 0.4



# Usage


## 机器翻译 translation

For 
```sh
$ export PYTHONPATH=.
$ python seq2seq/seq2seq_train.py --task translation
```


Attention is all your needs.
```sh
$ python seq2seq_train.py -- task translation --model=transformer
```

中英文翻译
```sh
$ python seq2seq_train.py -- task translation --model=transformer --dataset=en-zh
```

## 聊天对话 QA

数据格式：
```
question answer
```

QA和翻译，区别仅仅是qa的输入输出共用词典，模型通用

```
$ python seq2seq/seq2seq_train.py --task qa

```

## 基于方面的情感分析 

aspect-level sentiment classification

数据格式：
```
sentent, aspect, label
```

Aspect-level sentiment classification
```
$
```

## 自然语言推理 NLI

数据格式：
```
sentent1, sentence2, label
```




推理，是一个pair句子，有label



## config

修改 config.py



## 句子不等长的策略

1. 对input句子进行zero-padding
zero表示什么？表示SOS吗？还是单独的一个东西？对应embedding吗？


1. 对hidden-output设置为zero。
计算到input_length就停止。这个对动态图可行，缺陷是不能batch，或者只能等length的batch。
比如pytorch tutorial的实现

1. 对超出input-length的attention设置为zero
没见过


## 扩展阅读

- https://github.com/tensorflow/tensor2tensor
T2T 有助于针对各种机器学习应用（如翻译、解析、图像字幕制作等）创建最先进的模型，从而以远胜于过去的速度探索各种想法。
- dataset https://github.com/pytorch/text/blob/master/torchtext/datasets/
- https://github.com/pytorch/tutorials/blob/master/intermediate_source/seq2seq_translation_tutorial.py
- https://github.com/google/seq2seq/tree/master/seq2seq/ 通用的seq2seq框架，可借鉴

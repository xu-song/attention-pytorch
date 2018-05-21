# -*- coding: utf-8 -*-

'''
pytorch 0.4新变动：
去掉 from torch.autograd import Variable，所有的variable都替换成了tensor。

1. tensor必须转化成variable才能够计算梯度，variable和tensor不能直接计算，必须转化为同一类型。
   比如y=wx+b+1，1必须转化成variable才行。
2. variable.data对应tensor。0.4之后直接是tensor计算了，不用.data了
'''

from __future__ import unicode_literals, print_function, division

import random
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils import *
from encoder import *
from decoder import *
from multitask_snli_eval import evaluateSet
from dataset.snli import *

######################################################################
# Parameter
# ------------------
dataset = 'lang'   # 'lang' 'xiaohuangji' 'aspect'
use_pretrain = False 
pretrained_weight = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


input_lang, output_lang, label, train_set, test_set = prepareData()


######################################################################
# Training the Model
# ------------------

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, label_tensor, encoder, decoder1, decoder2, encoder_optimizer, decoder1_optimizer, decoder2_optimizer, criterion, max_length=MAX_LENGTH, aspect_variable=None):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder1_optimizer.zero_grad()
    decoder2_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    premise_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    hypothesis_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
    loss = [0, 0] # loss for multi-task
    for ei in range(input_length):
        premise_i, premise_hidden = encoder(input_tensor[ei], encoder_hidden)  # 1. 用于生成hypothesis 2. 用于分类
        premise_outputs[ei] = premise_i[0,0]

    # 生成任务
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    is_correct = False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder1(
                decoder_input, decoder_hidden, encoder_outputs)
            loss[0] += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder1(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            
            loss[0] += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    # 分类任务
    for ei in range(target_length):
        hypothesis_i, hypothesis_hidden = encoder(target_tensor[ei], encoder_hidden)
        hypothesis_outputs[ei] = hypothesis_i[0,0]

    mean_weight1 = torch.ones(1, self.input_length)/self.input_length
    mean_weight2 = torch.ones(1, self.target_length)/self.target_length
    premise = torch.bmm(mean_weight1.unsqueeze(0), premise_outputs.unsqueeze(0))
    hypothesis = torch.bmm(mean_weight2.unsqueeze(0), hypothesis_outputs.unsqueeze(0))
    
    decoder2(hypothesis)
    
    decoder_output = decoder2(torch.cat([premise, hypothesis], 1))
    loss[1] += criterion(decoder_output, target_variable[0])
    
    # 统一优化，梯度下降                
    loss = loss[0] + loss[1]
    loss.backward()

    encoder_optimizer.step()
    decoder1_optimizer.step()
    decoder2_optimizer.step()

    return loss.item() / target_length, is_correct
 




def trainIters(encoder, decoder1, decoder2, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, save_every=1000):
    start = time.time()
    plot_losses = []
    plot_losses_test = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder1_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    decoder2_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    training_triples_raw = [random.choice(train_set) for i in range(n_iters)]
    training_triples = [[tensorFromSentence(input_lang, data[0]), 
        tensorFromSentence(aspect_lang, data[1], False), tensorFromLabel(label,data[2])]
        for data in training_triples_raw]
    
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        training_triple = training_triples[iter - 1]
        input_tensor = training_triple[0]
        target_tensor = training_triple[1]
        label_tensor = training_triple[2]

        loss = train(input_tensor, target_tensor, label_tensor, 
                     encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            test_loss_avg = evaluateSet(encoder, decoder, test_set, input_lang, output_lang, criterion)
            print('%s (%d %d%%) train-loss:%.4f test-loss:%.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg, test_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            test_loss_avg = evaluateSet(encoder, decoder, test_set, input_lang, output_lang, criterion)
            plot_losses_test.append(test_loss_avg)

        if iter % save_every ==0:
            torch.save(encoder.state_dict(), 'model/encoder')
            torch.save(decoder.state_dict(), 'model/decoder')



    #showPlot(plot_losses, plot_losses_test)




######################################################################
# Training
# =======================
hidden_size = 200

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)
softmax_decoder1 = DecoderSoftmax(hidden_size*2, label).to(device)


torch.save(input_lang, 'model/input_lang')
torch.save(output_lang, 'model/output_lang')
torch.save(test_set, 'model/test_set')

trainIters(encoder1, attn_decoder1, softmax_decoder1, 7500000,  print_every=5000, plot_every=5000, save_every=5000)
#trainIters(encoder1, attn_decoder1, 70,  print_every=10, plot_every=10, save_every=10)

torch.save(encoder1.state_dict(), 'model/encoder')
torch.save(attn_decoder1.state_dict(), 'model/decoder')





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


# 为了更通用，建议采用import attnencoder as encoder
from encoder import EncoderRNN
from decoder import AttnDecoderRNN
from seq2seq.seq2seq_eval import evaluateSet
from utils.util import *



import config
args = config.get_args()
print(args.task)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

config.makedirs(args.save_path)

if(args.task=="translation"):
    from dataset.translation import *
    if(args.dataset=='eng-fra'):
        input_lang, output_lang, pairs = prepareData('eng', 'fra', True, index=True)
        test_size = int(len(pairs)/6)
        
    random.shuffle(pairs)
    test_set = pairs[0:test_size]
    train_set = pairs[test_size:len(pairs)]
    print("Train pairs %s; Test pairs %s" % (len(pairs)-test_size, test_size))
    print(random.choice(pairs))
    
elif(args.task=="qa"):
    from dataset.xiaohuangji import *
    input_lang, output_lang, pairs = prepareData()
    test_size = 1000
 
    random.shuffle(pairs)
    test_set = pairs[0:test_size]
    train_set = pairs[test_size:len(pairs)]
    print("Train pairs %s; Test pairs %s" % (len(pairs)-test_size, test_size))
    print(random.choice(pairs))

elif(args.task=="snli"):
    from dataset.snli import *
    input_lang, output_lang, train_set, test_set = prepareData(label='entailment')



######################################################################
# Training the Model
# ------------------

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, aspect_variable=None):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0,0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    
    is_correct = False
    
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, save_every=1000):
    start = time.time()
    plot_losses = []
    plot_losses_test = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(train_set))
                      for i in range(n_iters)]  # 当n_iters很大时，这里会特别占内存，而且特别慢，需要优化。
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
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
            torch.save(encoder.state_dict(), args.save_path + '/encoder')
            torch.save(decoder.state_dict(), args.save_path + '/decoder')



    #showPlot(plot_losses, plot_losses_test)




######################################################################
# Training
# =======================
hidden_size = 200 

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)


torch.save(input_lang, args.save_path + '/input_lang')
torch.save(output_lang, args.save_path + '/output_lang')
torch.save(test_set, args.save_path + '/test_set')

print(args.print_every)
trainIters(encoder1, attn_decoder1, args.n_iters,  args.print_every, args.plot_every, save_every=args.save_every)

torch.save(encoder1.state_dict(), args.save_path + '/encoder')
torch.save(attn_decoder1.state_dict(), args.save_path + '/decoder')





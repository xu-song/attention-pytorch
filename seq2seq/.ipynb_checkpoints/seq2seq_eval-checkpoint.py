# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


from encoder import EncoderRNN
from decoder import *


hidden_size = 200 
#from dataset.translation import *  # 这样很不好，要么train之后给save下来
from dataset.snli import *

# get test set




######################################################################
# Load Model
# =================





######################################################################
# Plotting results
# ----------------

import matplotlib.pyplot as plt
plt.switch_backend('agg') # 为什么新添加这一行？
import matplotlib.ticker as ticker
import numpy as np



def showPlot(points, points2=None):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    if(points2):
        plt.plot(points2)
        plt.legend(['train-loss', 'test-loss'], loc='upper left')
    plt.savefig('loss.jpg')



def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH, target_sentence=None, criterion=None):
    if(target_sentence): 
        target_tensor = tensorFromSentence(output_lang, target_sentence)

    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        loss = 0
        num_loss = 0

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)

            if(target_sentence and di < len(target_tensor)):
                loss += criterion(decoder_output, target_tensor[di])
                num_loss += 1
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
        if(target_sentence):
            return decoded_words, decoder_attentions[:di + 1], loss.item() / num_loss
        else:
            return decoded_words, decoder_attentions[:di + 1]



######################################################################
# We can also evaluate on the test set

def evaluateSet(encoder, decoder, eval_set, input_lang, output_lang, criterion):
    loss_total = 0
    for i in range(len(eval_set)):
        pair = eval_set[i]
        output_words, attentions, loss = evaluate(encoder, decoder, pair[0], 
                                      input_lang, output_lang, target_sentence=pair[1], criterion=criterion)
        loss_total += loss
    return loss_total/len(eval_set)




######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder, decoder, eval_set, input_lang, output_lang, n=10):
    for i in range(n):
        pair = random.choice(eval_set)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')





######################################################################
# Visualizing Attention
# ---------------------

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig('attention.jpg')
    #plt.show()


def evaluateAndShowAttention(encoder1, attn_decoder1, input_lang, output_lang, input_sentence, showAttn=True):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    if showAttn:
        showAttention(input_sentence, output_words, attentions)






######################################################################
# Evaluation
# =======================

def eval():

    # 或者在train.py中import进来
    input_lang = torch.load('model/input_lang')
    output_lang = torch.load('model/output_lang')
    eval_set = torch.load('model/test_set')

    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)

    encoder1.load_state_dict(torch.load('model/encoder'))
    attn_decoder1.load_state_dict(torch.load('model/decoder'))


    evaluateRandomly(encoder1, attn_decoder1, eval_set, input_lang, output_lang)
    evaluateAndShowAttention(encoder1, attn_decoder1, input_lang, output_lang, random.choice(eval_set)[0])

    try:
        while True:
            in_text = input("input " + input_lang.name + ":  press ctrl+c to exit\n")
            evaluateAndShowAttention(in_text, False)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    eval()
    print("Done")
#evaluateAndShowAttention("elle a cinq ans de moins que moi .")

#evaluateAndShowAttention("elle est trop petit .")

#evaluateAndShowAttention("je ne crains pas de mourir .")

#evaluateAndShowAttention("c est un jeune directeur plein de talent .")
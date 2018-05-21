# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################
# The Decoder
# -----------


######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
#
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#
# .. figure:: /_static/img/seq-seq-images/decoder-network.png
#    :alt:
#
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)




######################################################################
# aspect-level classification 也可以视为广义的Encoder-Deocder架构
# 其中，分类器可以视为的Attention-Decoder.
# translation中需要对每个output-word作attention，等价于这里对每个aspect做attention
# 区别是，translation中的Decoder是LSTM(输出是sequence)，并且以regression loss作为目标。这里的Decoder的输出是一个类别(给定aspect)，优化目标是classification loss。

# softmax as decoder, to decode a single class, not sequence
class DecoderSoftmax(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderSoftmax, self).__init__()
        self.hidden_size = hidden_size

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    # last output
    def forward(self, last_output, encoder_outputs = None):

        # self attention

        # commmon
        output = self.softmax(last_output)
        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)





class DecoderAspect(nn.Module):
    def __init__(self, hidden_size, aspect_size, output_size, max_length, dropout_p=0.1):
        '''
        output_size: class num
        '''
        super(DecoderAspect, self).__init__()
        self.hidden_size = hidden_size
        self.aspect_size = aspect_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # aspect embedding
        self.aspect_embedding = nn.Embedding(aspect_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.attn_linear1 = nn.Linear(self.hidden_size, self.max_length)
        self.attn_linear2 = nn.Linear(self.hidden_size * 2, self.max_length)
        self.w_h = torch.tensor(torch.randn(self.hidden_size, self.hidden_size), requires_grad=True)
        self.w_v = torch.tensor(torch.randn(self.hidden_size, self.hidden_size), requires_grad=True)
        self.w = torch.tensor(torch.randn(self.hidden_size, 1), requires_grad=True)
        self.w2 = torch.tensor(torch.randn(self.hidden_size*2, 1), requires_grad=True)

    def attn_weight(self, aspect, encoder_outputs):
        input_length = encoder_outputs.shape[0]
        attn_method = "mean pooling"
        if(attn_method=="atae"):
            M = torch.nn.Tanh()(torch.cat((aspect.repeat(input_length,1).mm(self.w_v), encoder_outputs.mm(self.w_h)), 1))
            attn = M.mm(self.w2).transpose(0,1)
            return F.softmax(attn, dim=1)


        if(attn_method=="self attention"):
            M = torch.nn.Tanh()(encoder_outputs.mm(self.w_h))
            attn = M.mm(self.w).transpose(0,1)
            return F.softmax(attn, dim=1)


        # mean pooling
        if(attn_method == "mean pooling"):
            return torch.tensor(torch.ones(1, self.max_length)/self.max_length, requires_grad=False)


        return None

    # aspect as input, no previous hidden. 因为decoder生成的不是sequence，所以没有previous output
    def forward(self, aspect, encoder_outputs):
        embedded = self.aspect_embedding(aspect).view(1, 1, -1)
        attn_weights = self.attn_weight(embedded[0], encoder_outputs)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = attn_applied
        output = self.softmax(self.out(output[0]))
        return output, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^
#
# If only the context vector is passed betweeen the encoder and decoder,
# that single vector carries the burden of encoding the entire sentence.
#
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs.

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        #self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        self.attn_linear1 = nn.Linear(self.hidden_size, self.max_length)
        self.attn_linear2 = nn.Linear(self.hidden_size * 2, self.max_length)
        self.w_h = torch.tensor(torch.randn(self.hidden_size, self.hidden_size), requires_grad=True)
        self.w_v = torch.tensor(torch.randn(self.hidden_size, self.hidden_size), requires_grad=True)
        self.w = torch.tensor(torch.randn(self.hidden_size, 1), requires_grad=True)
        self.w2 = torch.tensor(torch.randn(self.hidden_size*2, 1), requires_grad=True)


        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def attn_weight(self, previous_output, previous_hidden, encoder_outputs):
        attn_method = "default"

        if(attn_method=="default"):
            attn = self.attn_linear2(torch.cat((previous_output, previous_hidden), 1))
            return F.softmax(attn, dim=1)

        # embed and hidden is highly correlated, no need to use both
        if(attn_method=="default_sub_1"):
            attn = self.attn_linear1(previous_hidden)
            return F.softmax(attn, dim=1)

        if(attn_method=="default_sub_2"):
            attn = self.attn_linear1(previous_output)
            return F.softmax(attn, dim=1)

        # attention weight in "Attention-based LSTM for Aspect-level Sentiment Classification"
        if(attn_method=="atae"):
            M = torch.nn.Tanh()(torch.cat((previous_hidden.repeat(self.max_length,1).mm(self.w_v), encoder_outputs.mm(self.w_h)), 1))
            attn = M.mm(self.w2).transpose(0,1)
            return F.softmax(attn, dim=1)

        # self attention
        if(attn_method=="self attention"):
            M = torch.nn.Tanh()(encoder_outputs.mm(self.w_h))
            attn = M.mm(self.w).transpose(0,1)
            return F.softmax(attn, dim=1)



        # mean pooling
        if(attn_method == "mean pooling"):
            return Variable(torch.ones(1, 10)/10, requires_grad=False)

        return None


    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = self.attn_weight(embedded[0], hidden[0], encoder_outputs)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = torch.zeros(1, 1, self.hidden_size)
        if use_cuda:
            return result.cuda()
        else:
            return result






    

######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

def evaluate(encoder, decoder, sentence, aspect, label_name, criterion=None, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    aspect_variable = variableFromSentence(aspect_lang, aspect, False)
    target_variable = variableFromLabel(label, label_name)

    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    decoder_output, decoder_attention = decoder(aspect_variable, encoder_outputs)
    if criterion is not None:
        loss= criterion(decoder_output, target_variable[0])

    topv, topi = decoder_output.data.topk(1)
    ni = topi[0][0]
    
    if criterion is not None:
        return ni, decoder_attention.data, loss[0]
    else:
        return ni, decoder_attention.data

    
    
def evaluateRandomly(encoder, decoder, n=20):
    for i in range(n):
        triple = random.choice(test_set)
        print('>', triple[0])
        print('>', triple[1])
        print('=', triple[2])
        label_index, attn_weight = evaluateAspect(encoder, decoder, triple[0], triple[1], triple[2])
        label_name = label.labels[label_index]
        print('<', label_name)
        print('')
        

def evaluateSetAspect(encoder, decoder, criterion):
    loss_total = 0
    num_correct = 0
    for i in range(len(test_set)):
        triple = test_set[i]
        label_index, attn_weight, loss = evaluateAspect(encoder, decoder, triple[0], triple[1], triple[2], criterion=criterion)
        true_index = label.label2index[triple[2]]
        if label_index == true_index:
            num_correct += 1
        loss_total += loss
    return loss_total.data[0]/len(test_set), num_correct/len(test_set)





input_lang = torch.load('model/input_lang')
aspect_lang = torch.load('model/aspect_lang')
label = torch.load('model/label')

encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = DecoderAspect(hidden_size, aspect_lang.n_words, len(label.labels), MAX_LENGTH)

encoder1.load_state_dict(torch.load('model/encoder'))
attn_decoder1.load_state_dict(torch.load('model/decoder'))


######################################################################
#

evaluateRandomly(encoder1, attn_decoder1)
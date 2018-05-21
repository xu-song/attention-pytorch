


input_lang, aspect_lang, label, train_set, test_set = prepareDataAspect(index=is_train)
random.shuffle(train_set)
output_lang = aspect_lang



teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, aspect_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # 或者判断decoder
    is_correct = False

    decoder_output, decoder_attention = decoder(aspect_variable, encoder_outputs)
    loss += criterion(decoder_output, target_variable[0])

    #l2_reg = Variable( torch.FloatTensor(1), requires_grad=True)
    #for W in encoder.parameters():
    #    l2_reg = l2_reg + W.norm(2)
    #for W in decoder.parameters():
    #    l2_reg = l2_reg + W.norm(2)
    #print('%.3f, %.3f' % (loss,l2_reg))
    #loss += 0.000001*l2_reg

    topv, topi = decoder_output.data.topk(1)
    ni = topi[0][0]
    if ni == target_variable[0].data.numpy()[0]:
        is_correct = True
    #else:
    #    print('truth-label: %s, predict-label: %s' % (label.labels[target_variable[0].data.numpy()[0]], label.labels[ni]))




    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    if type(decoder) is DecoderAspect:
        return loss.data[0] / target_length, is_correct
    else:
        return loss.data[0] / target_length


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, save_every=1000):
    start = time.time()
    plot_losses = []
    plot_losses_test = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    num_correct = 0
    print_acc_total = []
    plot_acc_total = []

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_triples_raw = [random.choice(train_set) for i in range(n_iters)]
    training_triples = [[variableFromSentence(input_lang, data[0]), 
        variableFromSentence(aspect_lang, data[1], False), variableFromLabel(label,data[2])]
        for data in training_triples_raw]
    
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_triple = training_triples[iter - 1]
        input_tensor = training_triple[0]
        aspect_tensor = training_triple[1]
        target_tensor = training_triple[2]

        loss, is_correct = train(input_tensor, target_tensor, aspect_tensor, 
                     encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        if is_correct: 
            num_correct += 1
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            train_accuracy = num_correct / print_every
            num_correct = 0
            test_loss_avg, test_accuracy = evaluateSetAspect(encoder, decoder, criterion)
            print('%s (%d %d%%) train-loss:%.4f train-acc:%.4f test-loss:%.4f test-acc:%.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg, train_accuracy, test_loss_avg, test_accuracy))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            test_loss_avg, test_accuracy = evaluateSetAspect(encoder, decoder, criterion)
            plot_losses_test.append(test_loss_avg)

        if iter % save_every ==0:
            torch.save(encoder.state_dict(), 'model/encoder')
            torch.save(decoder.state_dict(), 'model/decoder')



    showPlot(plot_losses, plot_losses_test)
    
    
    


######################################################################
# Training and Evaluating
# =======================
hidden_size = 200 

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda
if is_train:

    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, pretrained_weight)
    attn_decoder1 = DecoderAspect(hidden_size, aspect_lang.n_words, len(label.labels), MAX_LENGTH)

    torch.save(input_lang, 'model/input_lang')
    torch.save(output_lang, 'model/aspect_lang')
    torch.save(label, 'model/label')
    trainItersAspect(encoder1, attn_decoder1, 70, print_every=10, plot_every=100000000, save_every=1000)

    torch.save(encoder1.state_dict(), 'model/encoder')
    torch.save(attn_decoder1.state_dict(), 'model/decoder')
    


import logging
import random
import time

import torch.optim as optim

from Translation import util
from Translation.model import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


def prepare_data(lang1_name, lang2_name, filename, reverse=False):
    input_lang, output_lang, pairs = util.read_langs(lang1_name, lang2_name, filename, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = util.filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepare_data('fra', 'eng', 'fra.txt')

# Print an example pair
print(random.choice(pairs))


def train(encoder, decoder, input_words, target_words, encoder_optimizer, decoder_optimizer, criterion,
    teacher_forcing_ratio, clip):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_len = input_words.size()[0]
    target_len = target_words.size()[0]
    loss = 0

    encoder_hidden = encoder.init_hidden()
    encoder_out, encoder_hidden = encoder(input_words, encoder_hidden)

    # 对于decoder 初始需要context, words_input, hidden
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))  # 用全0构建context
    decoder_input = Variable(torch.LongTensor([util.SOS_token]))  # 用开始标志作为初始输入
    decoder_hidden = encoder_hidden  # 使用encoder hidden 作为hidden

    use_teacher_force = random.random() < teacher_forcing_ratio

    if use_teacher_force:

        for i in range(target_len):  # decode 出 target_len 个单词
            decoder_out, decoder_context, decoder_hidden, atten_score = decoder(decoder_input, decoder_hidden,
                                                                                encoder_out, decoder_context)

            loss += criterion(decoder_out, target_words[i])

            decoder_input = target_words[i]

    else:
        for i in range(target_len):  # decode 出 target_len 个单词
            decoder_out, decoder_context, decoder_hidden, atten_score = decoder(decoder_input, decoder_hidden,
                                                                                encoder_out, decoder_context)

            loss += criterion(decoder_out, target_words[i])  # 这里decoder_out 必须是二维或者四维向量

            top_value, top_index = decoder_out.data.topk(1)

            next_input = top_index[0][0]  # why
            decoder_input = Variable(torch.LongTensor([next_input]))

            if next_input == util.EOS_token: break

    loss.backward()
    # 为了防止梯度爆炸，clip norm
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_len  # why use data[0]


def init_from_scrach():
    attn_model = 'general'
    hidden_size = 500
    n_layers = 2
    dropout_p = 0.05

    learning_rate = 0.0001

    encoder = RNNEncoder(hidden_size, input_lang.n_words, n_layers=n_layers)
    encoder_optim = optim.Adam(encoder.parameters(), lr=learning_rate)

    decoder = AttenedDecoderRnn(attn_model, hidden_size, output_lang.n_words, n_layers=n_layers, dropout_p=dropout_p)
    decoder_optim = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    return encoder, encoder_optim, decoder, decoder_optim, criterion


def main():
    # Configuring training
    n_epochs = 50000
    plot_every = 200
    print_every = 1000
    teacher_forcing_ratio = 0.5
    clip = 5.0

    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder, encoder_optim, decoder, decoder_optim, criterion = init_from_scrach()

    for epoch in range(1, n_epochs + 1):

        train_sample = util.variables_from_pair(random.choice(pairs), input_lang, output_lang)

        loss = train(encoder, decoder, train_sample[0], train_sample[1], encoder_optim, decoder_optim, criterion,
                     teacher_forcing_ratio, clip=clip)

        print_loss_total += loss
        plot_loss_total += loss

        if epoch == 0: continue

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (
                util.time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    show_plot(plot_losses)


def evaluation(sentence, encoder, decoder, max_length=util.MAX_LEN):
    input_variable = util.variable_from_sentence(input_lang, sentence)
    input_length = input_variable.size()[0]

    encoder_hidden = encoder.init_hidden()
    encoder_out, encoder_hidden = encoder(input_variable, encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[util.SOS_token]]))
    decoder_hidden = encoder_hidden
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))

    decoded_words = []

    # attention 是指当前words，对于encoder输出的每一个单词的表示的"重要性"，所以它的维度应该是 1 * B * seq_len
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_out, decoder_context, decoder_hidden, atten_score = decoder(decoder_input, decoder_hidden,
                                                                            encoder_out, decoder_context)
        top_value, top_index = decoder_out.data.topk(1)

        next_input = top_index[0][0]

        if next_input != util.SOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[next_input])

        decoder_input = Variable(torch.LongTensor([[next_input]]))

        decoder_attentions[di, :atten_score.size(2)] = atten_score

    return decoded_words, decoder_attentions


def evaluate_random(encoder, decoder):
    evaluation = random.choice(pairs)

    output_words, atten_scores = evaluation(evaluation[0], encoder, decoder)

    output_sentence = ' '.join(output_words)


    print('>', evaluation[0])
    print('=', evaluation[1])
    print('<', output_sentence)
    print('')


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


if __name__ == '__main__':
    main()

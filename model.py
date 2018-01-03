import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
全文整个模型的组成（att选择general时）
RNNEncoder (
  (embedding): Embedding(10, 10)
  (gru): GRU(10, 10, num_layers=2)
)
AttenedDecoderRnn (
  (embedding): Embedding(10, 10)
  (gru): GRU(20, 10, num_layers=2, bias=0.1) # 这里的输入为(word_emb, context)
  (atten): GeneralAttn (
    (att): Linear (10 -> 10)
  )
  (fc1): Linear (20 -> 10)
)

encoder层，输入words，(10, 10)，一共10个单词，每个词10维
    1.进入embedding -> embeded
    2.进入gru   -> outputs (sqe_len * B * hidden_size)， hidden -> ((layers * directions) * B * hidden_size)
    
    设置gru的目的是为了使当前句子里面的内容 互相关联 产生embedding
    
decoder层，输入上一个时序状态的word，也就是翻译好的word和hidden
    1. embedding层
    2.拼接[word_embed, last_context],输入gru，计算出当前词
    3.进入attention层，该层的输入包括encoder_outputs, decoder_last_hidden
        它的计算方式是，对当前的一个hidden，对每一个encoder_output进行一个函数f的计算，然后当做能量方程的各个维度，计算出一个当前hidden
        对每一个encoder_output的重要程度weight(seq_len * 1)。然后求出一个context=\sum weight_i * h_i，也就是一个单词hidden_size的维度
    4.encoder_output.bmm(atten_score)
    5.拼接[hidden, context], 经过fc，计算out_put
    6.soft-max

decoder 层相当于解压层，解压上下文语义，但是它并没有原文的上下文语义，所以这里加了对encoder的一个context，当做是上下文语义。
'''


class RNNEncoder(nn.Module):
    '''
    encoder 基于word的，使用一个gru，输出Out,和包含之前所有状态的seq
    '''

    def __init__(self, emb_dim, word_num, n_layers=1):
        super(RNNEncoder, self).__init__()
        self.embedding = nn.Embedding(word_num, embedding_dim=emb_dim)

        self.input_size = word_num
        self.hidden_size = emb_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(emb_dim, emb_dim, num_layers=n_layers)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        # if USE_CUDA: hidden = hidden.cuda()
        return hidden

    def forward(self, x, hidden):
        seq_len = len(x)

        embs = self.embedding(x).view(seq_len, 1, -1)

        out, hidden = self.gru(embs, hidden)

        return out, hidden


class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()

        self.drop_out = dropout_p
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.drou_out_layer = nn.Dropout(dropout_p)
        self.attn = GeneralAttn(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, words_input, last_hidden, encoder_output):
        '''
        words_input: 1
        last_hidden: 1 * B * hidden_size
        encoder_output : seq_len * B * hidden_size

        基于ATT的一个RNN，
            1. 首先计算embedding
            2. 然后计算对于encoder的输出单词
            3. 根据能量函数，求RNN上一个hidden输出的，依次跟encoder的每一个输出(word)的一个函数映射(随意)
            4. softmax输出一个权重值，应当是一个常数，然后对encoder的每一个输出做乘积，然后求和，这个维度应该为: 1 * B * hidden_size
            5. att之后的context表示上一层的RNN的hidden对encoder的每一个输出的重要程度。
            6. 结合当前将要送入RNN的emb，送进RNN
        所以呢，ATT求出来的是对当前输入的word的
        :param words_input:
        :param last_hidden:
        :param encoder_output:
        :return:
        '''
        embs = self.embedding(words_input).view(1, 1, -1)  # words_input: 1 * B * hidden_size

        embs_droped = self.drou_out_layer(embs)

        atten_weight = self.attn(last_hidden[-1], encoder_output)  # 这里-1可以不加，一样的

        context = atten_weight.bmm(encoder_output.transpose(0, 1))

        rnn_in = torch.cat((embs_droped, context), 2)

        out, hidden = self.gru(rnn_in, last_hidden)  # 1 * B * N 表示预测的输出

        output = out.squeeze(0)  # B x N
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))

        return output, hidden, atten_weight


class GeneralAttn(nn.Module):
    def __init__(self, method, hidden_size, n_layers=1, dropout_p=0.1):
        super(GeneralAttn, self).__init__()

        self.drop_out = dropout_p
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.method = method

        if self.method == 'general':
            self.att = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':  # concat 之后应该是一个列向量
            self.att = nn.Linear(self.hidden_size * 2, self.hidden_size)  # y = w * x
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))
        elif self.method == 'dot':
            pass
        else:
            raise Exception('No %s type attention error' % self.method)

    def forward(self, encoder_outputs, hidden):
        '''
        encoder_output ： seq_len * B * hidden_size  每个都是列向量
        hidden : (n_direction *n_layer) * B * hidden_size   行向量
        :param encoder_output:
        :param hidden:
        :return:
        '''
        seq_len = len(encoder_outputs)

        att_scores = Variable(torch.zeros(seq_len))  # 正常情况下应该是 seq_len * B * 1

        for i in range(len(encoder_outputs)):
            att_scores[i] = self.score(encoder_outputs[i], hidden)  # seq_len * 1 * 1

        return F.softmax(att_scores).unsqueeze(0).unsqueeze(0)

    def score(self, encoder_output, hidden):
        if self.method == 'dot':
            return hidden.dot(encoder_output)
        elif self.method == 'general':
            attened = self.att(encoder_output)
            return hidden.view(-1).dot(attened.view(-1))
        else:
            attened = self.att(torch.cat((encoder_output, hidden), 1))
            return self.other.view(-1).dot(attened.view(-1))


class AttenedDecoderRnn(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, dropout_p=0.1, n_layers=1):
        super(AttenedDecoderRnn, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, hidden_size)

        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout_p)
        if self.attn_model is not None:
            self.atten = GeneralAttn(self.attn_model, hidden_size)

        self.fc1 = nn.Linear(hidden_size * 2, output_size)

    def forward(self, word_input, decoder_last_hidden, encoder_outputs, last_context):
        '''
        预测下一个单词的概率
            decoder output value y is the embeded word which updata by last time

            context 表示的是，对于上一轮的输出，我们应该在哪一个encoder_output上集中精力，其输出是一个hidden_size的维度
            rnn_output 预测下一轮的单词，在没有atten的情况下
            rnn_hidden 这一轮rnn_hidden的输出

        :param word_input:
        :param decoder_last_hidden:
        :param encoder_outputs:
        :param last_context:
        :return:
        '''
        word_embed = self.embedding(word_input).view(1, 1, -1)

        rnn_input = torch.cat((word_embed, last_context.unsqueeze(0)), 2)  # 1 * 1 * (hidden_size * 2)
        rnn_output, rnn_hidden = self.gru(rnn_input,
                                          decoder_last_hidden)  # 1 * 1 * hidden_size,   (1 * 2) * 1 * hidden_size

        atten_socres = self.atten(encoder_outputs, rnn_output.squeeze(0))  # 这里有问题啊，为什么使用输出，而不是Hidden

        context = atten_socres.bmm(encoder_outputs.transpose(0, 1))  # (1 * 1 * 3) (3 * 1 * 10) = (1 * 1 * 10)

        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N

        context = context.squeeze(1)
        output = F.log_softmax(self.fc1(torch.cat((rnn_output, context), 1)))

        return output, context, rnn_hidden, atten_socres

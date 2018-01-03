from Translation.model import *
from torch.autograd import Variable
import torch

encoder_test = RNNEncoder(10, 10, 2)
decoder_test = AttenedDecoderRnn('general', 10, 10, n_layers=2)

word_input = Variable(torch.LongTensor([1, 2, 3]))
encoder_test.hidden = encoder_test.init_hidden()
encoder_outputs, encoder_test.hidden = encoder_test(word_input, encoder_test.hidden)

decoder_word_in = Variable(torch.LongTensor([1, 2, 3]))
context = Variable(torch.zeros(1, decoder_test.hidden_size))
decoder_attens = torch.zeros(1, 3, 3)
decoder_hidden = encoder_test.hidden # 为什么？

print(encoder_test)
print(decoder_test)

for i in range(len(decoder_word_in)):
    decoder_output, context, decoder_hidden, decoder_atten = decoder_test(decoder_word_in[i], decoder_hidden, encoder_outputs, context)
    print(decoder_output.size(), decoder_hidden.size(), decoder_atten.size())
    decoder_attens[0, i] = decoder_atten.squeeze(0).cpu().data

'''
torch.Size([1, 10]) torch.Size([2, 1, 10]) torch.Size([1, 1, 3])
torch.Size([1, 10]) torch.Size([2, 1, 10]) torch.Size([1, 1, 3])
torch.Size([1, 10]) torch.Size([2, 1, 10]) torch.Size([1, 1, 3])
'''




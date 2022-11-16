from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import sys
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
SOS_token = 0 # Start of sentence
EOS_token = 1 # End of sentence

class Lang: # This a class included all words
    def __init__(self, name):
        self.name = name
        self.word2index = {} # i.e {'hello':53}
        self.word2count = {} # How many a word repated in data like {'hello':21}
        self.index2word = {0: "SOS", 1: "EOS"} # i.e {944: 'cowards}'
        self.n_words = 2 # How many words exist in dataset?

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
#%% 
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
#%%
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
#%%
v = open('data/%s-%s.txt' % ('eng', 'fra'), encoding='utf-8').read().strip().split('\n')
vv = [[normalizeString(s) for s in l.split('\t')] for l in lines]
p2 = [list(reversed(p)) for p in vv]
p2[55]
#%%  
def readLangs(lang1, lang2, reverse=False):
    global lines; global pairs1; global pairs2
    # Read dataset from dir and split each line to a list like that =>  ['Really?\tVrai ?', 'We won.\tNous gagnâmes.', ...]
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    # Now, Turn like that [['really ?', 'vrai ?'], ['we won .', 'nous gagnames .'], ...]
    pairs1 = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse:
        # reverse pairs1 list like that [['vrai ?', 'really ?'], ['nous gagnames .', 'we won .'], ...]
        pairs2 = [list(reversed(p)) for p in pairs1]
        input_lang = Lang(lang2) # Pass eng to Lang class and get object (The object have no words in itself yet, Just a object)
        output_lang = Lang(lang1) # Pass fra to Lang class and get object (The object have no words in itself yet, Just a object)
    else:
        input_lang = Lang(lang1) 
        output_lang = Lang(lang2) 

    return input_lang, output_lang, pairs2 # pairs2 [['vrai ?', 'really ?'], ['nous gagnames .', 'we won .'], ...]
#%%
MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p): # Only setence with maximum 10 words and turn startswith with eng_prefixes
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    # lang1='eng' , lang2='fra'
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse) # pairs means all data in a list [['a','b'], ['c','d'], ''']
    pairs = filterPairs(pairs) # filter only setence with maximum 10 words and turn startswith with eng_prefixes
    for pair in pairs:
        input_lang.addSentence(pair[0]) # add fra setentence to input_lang
        output_lang.addSentence(pair[1]) # add eng setentence to output_lang
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
#%%
input_lang.word2index
# embedding = nn.Embedding(4345, 256)
# em = embedding(it).view(1, 1, -1)
# h = torch.zeros(1, 1, 256, device=device)
# gru = nn.GRU(256, 256)
# encoder_output, encoder_hidden = gru(em, h)
# v = torch.Tensor([[[1,1,1,1],[2,1,1,1],[3,1,1,1]]])
# v[1].size()
#%%
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        # hidden_size == 256
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size) # 4345,256
        self.gru = nn.GRU(hidden_size, hidden_size) # 256,256

    def forward(self, input, hidden):
        # input.size() => 1 at first place
        embedded = self.embedding(input).view(1, 1, -1) # embedding(input).size() 1,256 | embedding(input).view(1, 1, -1).size() 1,1,256
        output = embedded
        output, hidden = self.gru(output, hidden) # output.size() 1,1,256 | hidden.size() 1,1,256
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
#%%
# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, input, hidden):
#         output = self.embedding(input).view(1, 1, -1)
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)
#%%    
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        # hidden_size == 256 | output_size == 2803
        self.hidden_size = hidden_size # 256
        self.output_size = output_size # 2803
        self.dropout_p = dropout_p
        self.max_length = max_length # 10
        self.embedding = nn.Embedding(self.output_size, self.hidden_size) # 2803,256
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length) # 512,10
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size) # 512,256
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size) # 256,256
        self.out = nn.Linear(self.hidden_size, self.output_size) # 256,2803

    def forward(self, input, hidden, encoder_outputs):
        # input.size() => 1
        embedded = self.embedding(input).view(1, 1, -1) # embedding(input).size() 1,256 | embedding(input).view(1, 1, -1) 1,1,256
        embedded = self.dropout(embedded) # 1,1,256
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1) # embedded[0] 1,256 | hidden[0] 1,256 | 1,10
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)) # (1,1,10, 1,1,256) 1,1,256
        output = torch.cat((embedded[0], attn_applied[0]), 1) # (1,256, 1,256) | 1,512
        output = self.attn_combine(output).unsqueeze(0) # 1,1,256
        output = F.relu(output) # 1,1,256
        output, hidden = self.gru(output, hidden) # 1,1,256
        output = F.log_softmax(self.out(output[0]), dim=1) # 1,2803 
        return output, hidden, attn_weights # 1,2803, 1,1,256, 1,10

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
#%%  
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token) # add number 1
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
#%%
tensorsFromPair(pairs[50])

#%%
teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    global decoder_input; global decoder_output; global ta
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0) # Length of input sentence 
    target_length = target_tensor.size(0) # Length of output sentence 
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device) # 10,256
    
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden) # 1,1,256 | 1,1,256
        encoder_outputs[ei] = encoder_output[0, 0] # It's means all value in encoder_output
    
    decoder_input = torch.tensor([[SOS_token]], device=device) # tensor([[0]])
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs) # 1,2803 | 1,1,256 | 1,10
            # sys.exit()
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length
#%%
import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
#%%
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    # n_iters = 7500
    global input_tensor; global target_tensor; global training_pair
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0 

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # fetch 7500 pair randomly and then pass to tensorsFromPair to transform to tensor
    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)] 
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print(loss)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
#%%   
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]
#%%   
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
#%%        
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

evaluateRandomly(encoder1, attn_decoder1)
#%%
output_lang.word2count


import gzip
import io
import pickle
import re

import unicodedata
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from operator import mul

SOS_token = 0
EOS_token = 1


class RNN:
    def __init__(self, encoder_x_dim, decoder_x_dim, h_dim, epoch, learning_rate):
        self.encoder_x_dim = encoder_x_dim
        self.decoder_x_dim = decoder_x_dim
        self.h_dim = h_dim
        self.epoch = epoch
        self.learning_rate = learning_rate

        self.enc_wxh = np.random.randn(self.h_dim, self.h_dim) / 1000
        self.enc_whh = np.random.randn(self.h_dim, self.h_dim) / 1000
        self.enc_why = np.random.randn(self.encoder_x_dim, self.h_dim) / 1000
        self.enc_bh = np.random.randn(self.h_dim, 1) / 1000
        self.enc_by = np.random.randn(self.encoder_x_dim, 1)

        self.dec_wxh = np.random.randn(self.h_dim, self.h_dim) / 1000
        self.dec_whh = np.random.randn(self.h_dim, self.h_dim) / 1000
        self.dec_why = np.random.randn(self.decoder_x_dim, self.h_dim) / 1000
        self.dec_bh = np.random.randn(self.h_dim, 1) / 1000
        self.dec_by = np.random.randn(self.decoder_x_dim, 1)

    def softmax(self, x, derivative=False):
        x_safe = x + 1e-12
        f = np.exp(x_safe) / np.sum(np.exp(x_safe))

        if derivative:  # Return the derivative of the function evaluated at x
            pass  # We will not need this one
        else:  # Return the forward pass of the function at x
            return f
        return self

    def encoder_forward(self, inputs, embedding_matrix):
        outputs, hidden_states = [], []
        hidden_state = np.zeros((self.enc_whh.shape[0], 1))
        for record in inputs:
            embedding_layer = embedding_matrix.T.dot(record)
            hidden_state = np.tanh(self.enc_wxh.dot(embedding_layer) + self.enc_whh.dot(hidden_state) + self.enc_bh)
            # out = self.softmax(why.dot(hidden_state) + by)
            # outputs.append(out)
            hidden_states.append(hidden_state.copy())
        return outputs, hidden_states

    def decoder_forward(self, inputs, input_hidden_from_dec,
                        decod_embedding_matrix):
        net_outputs = []
        for record in inputs:
            embed_out = decod_embedding_matrix.T.dot(record)
            hs = np.tanh(self.dec_wxh.dot(embed_out) + self.dec_whh.dot(input_hidden_from_dec) + self.dec_bh)
            t_out = self.softmax(self.dec_why.dot(hs) + self.dec_by)
            net_outputs.append(t_out)
        return net_outputs

    def backward(self):
        return self

    def train(self, train_x, train_y, encoder_embedding_matrix, decoder_embedding_matrix):
        train_x = make_list(train_x)
        train_y = make_list(train_y)

        encoder_outputs, hidden_states = self.encoder_forward(train_x,
                                                              encoder_embedding_matrix)
        decoder_output = self.decoder_forward(train_y, hidden_states[-1],
                                              decoder_embedding_matrix)

        probs = [np.argmax(output) for output in decoder_output]

        self.backward(probs)

        return self


class VocabularyProperties:
    def __init__(self):
        self.word2idx = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.word2Count = {}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.word2Count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2Count[word] += 1


def read_embeddings(self, MAX_NUM_WORDS, word2idx_inputs, word2idx_output, EMBEDDING_SIZE, network_type):
    # embeding_dict = {}
    # glove_file = open('E:\\projects\\Source Code Summarization\\NMT_data\\glove.6B.100d.txt')
    # for line in glove_file:
    #     records = line.split()
    #     word = records[0]
    #     vector_dimensions = np.asarray(records[1:])
    #     embeding_dict[word] = vector_dimensions
    # glove_file.close()

    # with open('NMT_data\\embedding_dict.pickle', 'wb') as handle:
    #     pickle.dump(embeding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if network_type == 'encoder':
        with open('NMT_data\\encoder_embedding_dict.pickle', 'rb') as handle:
            embedding_dict = pickle.load(handle)
            widx = word2idx_inputs

        num_words = min(MAX_NUM_WORDS, len(widx) + 1)
        embedding_matrix = np.zeros((len(widx) + 1, 100))
        for word, index in widx.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
        return embedding_matrix

    else:
        fname = 'NMT_data\\decoder_embedding_dict.txt'
        widx = word2idx_output
        embedding_matrix = np.zeros((len(widx) + 1, 100))
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = tokens[1:]

        for word, idx in widx.items():
            myvec = data[word]
            if myvec is not None:
                embedding_matrix[idx] = myvec
        return embedding_matrix


def encode_sequences(tokenizer, length, lines, encoding_type):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding=encoding_type)
    return seq


class PrepareData:
    def __init__(self, data_address, english_to_deutch):
        self.data_address = data_address
        self.english_to_french = english_to_deutch

    def read_data(self):
        # open the file
        file = open(self.data_address, mode='rt', encoding='utf-8')
        # read all text
        text = file.read()
        file.close()
        return text

    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = s.replace('!', '')
        return s

    def iterate_data_lines(self, data):
        sents = data.strip().split('\n')
        sents = [[self.normalizeString(token) for token in item.split('  ')] for item in sents]
        return sents

    def tokenization(self, lines):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer, tokenizer.word_index

    def make_list(self, token_matrix):
        data = []
        for i in range(len(token_matrix)):
            rec = np.zeros((np.size(token_matrix, 1), 1))
            for j in range(len(token_matrix[i, :])):
                rec[j, 0] = token_matrix[i, j]
            data.append(rec)
        return data

    def indexesFromSentence(self, lang, sentence):
        return [lang.word2idx[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return indexes

    def tensorsFromPair(self, pair, input_lang_prop, target_lang_prob):
        input_tensor = self.tensorFromSentence(input_lang_prop, pair[0])
        target_tensor = self.tensorFromSentence(target_lang_prob, pair[1])
        return (input_tensor, target_tensor)

    def prepare_data(self):
        training_pairs = []
        input_data = self.read_data()
        sents = self.iterate_data_lines(input_data)

        input_lang_prop = VocabularyProperties()
        target_lang_prob = VocabularyProperties()

        for item in sents:
            input_lang_prop.addSentence(item[0])
            target_lang_prob.addSentence(item[1])

        for pair in sents:
            training_pairs.append(self.tensorsFromPair(pair, input_lang_prop, target_lang_prob))

        english, word2idx_input = self.tokenization(sents[:, 0])
        englich_vocab_size = len(english.word_index) + 1

        deu, word2idx_output = self.tokenization(sents[:, 1])
        deu_vocab_size = len(deu.word_index) + 1
        return self


def main():
    data_address = "NMT_data\\test_data.txt"
    english_to_deutch = True
    data_obj = PrepareData(data_address, english_to_deutch)
    text = data_obj.prepare_data()

    train, test = train_test_split(sents, test_size=0.2, random_state=12)

    encoder_input_data = encode_sequences(english, englich_vocab_size, train[:, 0], encoding_type='pre')
    decoder_input_data = encode_sequences(deu, deu_vocab_size, train[:, 1], encoding_type='post')
    decoder_test_data = encode_sequences(english, englich_vocab_size, test[:, 0], encoding_type='post')
    actual_test_data = encode_sequences(deu, deu_vocab_size, test[:, 1], encoding_type='post')

    encoder_x_dim = encoder_input_data.shape[1]
    decoder_x_dim = decoder_input_data.shape[1]
    h_dim = 100
    epoch = 10
    learning_rate = 0.0000009

    rnn = RNN(encoder_x_dim, decoder_x_dim, h_dim, epoch, learning_rate)

    # encoder_embedding_matrix = rnn.read_embeddings(len(sents), word2idx_input, word2idx_output, h_dim,
    #                                                network_type='encoder')
    # decoder_embedding_matrix = rnn.read_embeddings(len(sents), word2idx_input, word2idx_output, h_dim,
    #                                                network_type='decoder')

    with open('NMT_data\\encoder_embedding_matrix.pickle', 'rb') as handle:
        encoder_embedding_matrix = pickle.load(handle)

    with open('NMT_data\\decoder_embedding_matrix.pickle', 'rb') as handle:
        decoder_embedding_matrix = pickle.load(handle)
    rnn.train(encoder_input_data, decoder_input_data, encoder_embedding_matrix, decoder_embedding_matrix)


if __name__ == '__main__':
    lenc = LabelEncoder()
    enc = OneHotEncoder()
    main()

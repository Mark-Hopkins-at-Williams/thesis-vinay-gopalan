""" Script containing functions and neural networks for the bag-of-words and count-words models. """

import torch
import numpy as np

####################################################################################################################
# Common to both Bag-of-Words, Count-Words Models
####################################################################################################################

def simple_accuracy(preds, labels):
    """Takes in two lists of predicted labels and actual labels and returns the accuracy in the form of a float. """
    return np.equal(preds, labels).mean()

def get_labels(filename):
    """ Returns a list of labels for sentences in a given file. """
    with open(filename,'r') as reader:
        labels = []
        for line in reader:
            line_split = line.split('\t')
            if line_split[1].strip() != 'label' and len(line_split)>1:
                labels.append(int(line_split[1]))
            elif len(line_split) < 1:
                labels.append(0)
        return labels

def get_frequencies(filename):
    """ Returns a dictionary of the frequencies of all the words in a given file. """
    with open(filename,'r') as reader:
        frequencies = {}
        for line in reader:
            words_in_line = line.split('\t')[0].strip().split(' ')
            for word in words_in_line:
                if word in frequencies:
                    frequencies[word] += 1
                else:
                    frequencies[word] = 1
        return frequencies


def create_vocab(frequencies, K):
    """ Creates a list of words as vocabulary by selecting words with a frequency > K. """
    if '…' in frequencies:
        del frequencies['…']

    for key in list(frequencies):
        if frequencies[key] < K:
            del frequencies[key]
    return list(frequencies)


""" Models for setting baseline. """ 

def two_layer_feedforward(input_size, H):
    """
    A two-layer feedforward neural network with 'input_size' input features, H hidden
    features, and a softmax response value.
    
    """
    net = torch.nn.Sequential()
    net.add_module("dense1", torch.nn.Linear(in_features = input_size, 
                                   out_features = H))
    net.add_module("relu1", torch.nn.ReLU())
    net.add_module("dense2", torch.nn.Linear(in_features = H, 
                                   out_features = 3))
    net.add_module("softmax", torch.nn.Softmax(dim=1))

    return net


def three_layer_feedforward(input_size, H1, H2):
    """
    A three-layer feedforward neural network with 'input_size' input features, H1, H2 hidden
    features, and a softmax response value.
    
    """
    net = torch.nn.Sequential()
    net.add_module("dense1", torch.nn.Linear(in_features = input_size, 
                                   out_features = H1))
    net.add_module("relu1", torch.nn.ReLU())
    net.add_module("dense2", torch.nn.Linear(in_features = H1,
                                   out_features = H2))
    net.add_module("relu2", torch.nn.ReLU())
    net.add_module("dense3", torch.nn.Linear(in_features = H2,
                                   out_features = 3))
    net.add_module("softmax", torch.nn.Softmax(dim=1))

    return net


def four_layer_feedforward(input_size, H1, H2, H3):
    """
    A four-layer feedforward neural network with 'input_size' input features, H1, H2, H3 hidden
    features, and a softmax response value.
    
    """
    net = torch.nn.Sequential()
    net.add_module("dense1", torch.nn.Linear(in_features = input_size, 
                                   out_features = H1))
    net.add_module("relu1", torch.nn.ReLU())
    net.add_module("dense2", torch.nn.Linear(in_features = H1,
                                   out_features = H2))
    net.add_module("relu2", torch.nn.ReLU())
    net.add_module("dense3", torch.nn.Linear(in_features = H2,
                                   out_features = H3))
    net.add_module("relu3", torch.nn.ReLU())
    net.add_module("dense4", torch.nn.Linear(in_features = H3,
                                   out_features = 3))
    net.add_module("softmax", torch.nn.Softmax(dim=1))

    return net

####################################################################################################################
# Only used in Bag-of-Words model
####################################################################################################################

def create_vectors(filename, vocab):
    """ Creates 1-hot vectors from a given vocabulary set of words. """
    tweet_vectors = []
    hidden_length = len(vocab)
    with open(filename,'r') as reader:
        for line in reader:
            tweet_tensor = torch.zeros(hidden_length)
            words_in_line = line.split('\t')[0].strip().split(' ')
            if words_in_line[0] != 'sentence':
                for i in range(0,hidden_length):
                    if vocab[i] in words_in_line:
                        tweet_tensor[i] = 1
                tweet_vectors.append(tweet_tensor)
        return tweet_vectors


####################################################################################################################
# Only used in Count-Words model
####################################################################################################################


def get_word_frequencies(words):
    """ Returns a dictionary of the frequencies of all the words in a given line. """
    freq = {}
    for word in words:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    return freq

def create_count_vectors(filename, vocab):
    """ Creates vectors with word count from a given vocabulary set of words. """
    tweet_vectors = []
    hidden_length = len(vocab)
    with open(filename,'r') as reader:
        for line in reader:
            tweet_tensor = torch.zeros(hidden_length)
            words_in_line = line.split('\t')[0].strip().split(' ')
            line_frequencies = get_word_frequencies(words_in_line)
            if words_in_line[0] != 'sentence':
                for i in range(0,hidden_length):
                    if vocab[i] in words_in_line:
                        tweet_tensor[i] = line_frequencies[vocab[i]]
                tweet_vectors.append(tweet_tensor)
        return tweet_vectors


####################################################################################################################
# Only used in BiGrams
####################################################################################################################

def get_bigram_frequencies(filename):
    """ Returns a dictionary of the frequencies of all words and all bigrams in a given file. """
    with open(filename,'r') as reader:
        frequencies = {}
        for line in reader:
            words_in_line = line.split('\t')[0].strip().split(' ')
            for word in words_in_line:
                if word in frequencies:
                    frequencies[word] += 1
                else:
                    frequencies[word] = 1
            # Make BiGrams
            for i in range(1,len(words_in_line),2):
                if i != len(words_in_line) - 1:
                    bigram1 = words_in_line[i-1] + " " + words_in_line[i]
                    if bigram1 in frequencies:
                        frequencies[bigram1] += 1
                    else:
                        frequencies[bigram1] = 1
                    bigram2 = words_in_line[i] + " " +  words_in_line[i+1]
                    if bigram2 in frequencies:
                        frequencies[bigram2] += 1
                    else:
                        frequencies[bigram2] = 1
        return frequencies


def create_bigram_vectors(filename, vocab):
    """ Creates 1-hot vectors from a given vocabulary set of words. """
    tweet_vectors = []
    hidden_length = len(vocab)
    with open(filename,'r') as reader:
        for line in reader:
            tweet_tensor = torch.zeros(hidden_length)
            line = line.split('\t')[0].strip()
            if line != 'sentence':
                for i in range(0,hidden_length):
                    if vocab[i] in line:
                        tweet_tensor[i] = 1
                tweet_vectors.append(tweet_tensor)
        return tweet_vectors



####################################################################################################################
# Only used in TriGrams Extension
####################################################################################################################

def get_trigram_frequencies(filename):
    """ Returns a dictionary of the frequencies of all trigrams in a given file. """
    with open(filename,'r') as reader:
        frequencies = {}
        for line in reader:
            words_in_line = line.split('\t')[0].strip().split(' ')
            # Make TriGrams
            for i in range(2,len(words_in_line)-3,3):
                if i == len(words_in_line) - 1:
                    trigram1 = words_in_line[i-2] + " " + words_in_line[i-1] + " " + words_in_line[i]
                    if trigram1 in frequencies:
                        frequencies[trigram1] += 1
                    else:
                        frequencies[trigram1] = 1
                if i <= len(words_in_line) - 3:
                    trigram1 = words_in_line[i-2] + " " + words_in_line[i-1] + " " + words_in_line[i]
                    if trigram1 in frequencies:
                        frequencies[trigram1] += 1
                    else:
                        frequencies[trigram1] = 1
                    trigram2 = words_in_line[i] + " " +  words_in_line[i+1] + " " +  words_in_line[i+2]
                    if trigram2 in frequencies:
                        frequencies[trigram2] += 1
                    else:
                        frequencies[trigram2] = 1
        return frequencies
import torch
import numpy as np

"""Takes in two lists of predicted labels and actual labels and returns the accuracy in the form of a float. """
def simple_accuracy(preds, labels):
        return np.equal(preds, labels).mean()

""" Returns a list of labels for sentences in a given file. """
def get_labels(filename):
    with open(filename,'r') as reader:
        labels = []
        for line in reader:
            line_split = line.split('\t')
            if line_split[1].strip() != 'label' and len(line_split)>1:
                labels.append(int(line_split[1]))
            elif len(line_split) < 1:
                labels.append(0)
        return labels

""" Returns a dictionary of the frequencies of all the words in a given file. """
def get_frequencies(filename):
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

""" Creates a list of words as vocabulary by selecting words with a frequency > K. """
def create_vocab(frequencies, K):
    if '…' in frequencies:
        del frequencies['…']

    for key in list(frequencies):
        if frequencies[key] < K:
            del frequencies[key]
    return list(frequencies)

""" Creates 1-hot vectors from a given vocabulary set of words. """
def create_vectors(filename, vocab):
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


""" Model for setting baseline. """ 

def two_layer_feedforward(input_size, H):
    """
    A two-layer feedforward neural network with 'input_size' input features, H hidden
    features, and one softmax response value.
    
    """
    net = torch.nn.Sequential()
    net.add_module("dense1", torch.nn.Linear(in_features = input_size, 
                                   out_features = H))
    net.add_module("relu1", torch.nn.ReLU())
    net.add_module("dense2", torch.nn.Linear(in_features = H, 
                                   out_features = 3))
    net.add_module("softmax", torch.nn.Softmax(dim=1))

    return net

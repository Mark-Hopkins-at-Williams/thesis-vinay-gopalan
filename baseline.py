import torch

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

def create_vocab(frequencies, K):
    if '…' in frequencies:
        del frequencies['…']

    for key in list(frequencies):
        if frequencies[key] < K:
            del frequencies[key]
    return list(frequencies)

def create_vectors(filename, vocab):
    tweet_vectors = []
    hidden_length = len(vocab)
    with open(filename,'r') as reader:
        for line in reader:
            tweet_tensor = torch.zeros(hidden_length,requires_grad=True)
            words_in_line = line.split('\t')[0].strip().split(' ')
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
                                   out_features = 2))
    net.add_module("softmax", torch.nn.Softmax(dim=0))

    return net

def eval_results(filename):
    with open(filename,'w') as writer:
        frequencies = get_frequencies("data/SST-3/dev.tsv")
        vocabulary = create_vocab(frequencies, 15)
        inputs = create_vectors("data/SST-3/dev.tsv", vocabulary)
        input_size = len(inputs[0])
        print(input_size)
        net = two_layer_feedforward(input_size, 784)
        outs = [net(x) for x in inputs]
        for x in range(len(outs)):
            writer.write("%s %s\n"%(str(x),str(outs[x])))

if __name__ == "__main__":
   eval_results('base.txt')
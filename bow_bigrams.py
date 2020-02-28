""" The script to train and run the Bag-of-Words BiGrams extension on a trainset and devset. """

import torch
import numpy as np
import copy
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from baseline import get_bigram_frequencies, get_labels, create_vocab, simple_accuracy
from baseline import create_bigram_vectors
from baseline import two_layer_feedforward, three_layer_feedforward, four_layer_feedforward

##########################################################################################
# Important Constants

FREQ_THRESHOLD = 15
BATCH_SIZE = 6
VOCAB = []
INPUT_SIZE = 0
H1 = 300
H2 = 180
H3 = 75
NUM_EPOCHS = 20

##########################################################################################


def make_train_vectors(input_filename):
    """
    Builds training vectors from sentences in a given file by doing the following:
        1. Get the frequencies of all words in the file.
        2. Create a GLOBAL vocabulary using the frequencies and a threshold (Same vocab used by testing data).
        3. Create the vectors with vec[i] = 1 if vocab[i] in words_in_line. Else vec[i] = 0.
        4. Get the actual labels from the file.
        5. Return vecs, labels.
    """

    # Create vectors
    frequencies = get_bigram_frequencies(input_filename)
    global VOCAB, INPUT_SIZE
    VOCAB = create_vocab(frequencies, FREQ_THRESHOLD)
    vecs = create_bigram_vectors(input_filename, VOCAB)

    # Set value for input size
    INPUT_SIZE = len(VOCAB)

    # Get labels from file
    labels = get_labels(input_filename)

    assert(len(labels)==len(vecs))

    return vecs, labels

def make_test_vectors(input_filename):
    """
    Similar to make_train_vectors, except it only needs to create vectors using existing vocabulary.
    """ 

    # Create vectors
    vecs = create_bigram_vectors(input_filename, VOCAB)

    # Get labels from file
    labels = get_labels(input_filename)

    assert(len(labels)==len(vecs))

    return vecs, labels

class BiGramsTrainDataSet(Dataset):
    """ The BiGrams training dataset containing sentence vectors and actual labels. """
    def __init__(self, datafile):
        self.vecs, self.labels = make_train_vectors(datafile)
    
    def __len__(self):
        return len(self.vecs)

    def __getitem__(self, idx):
        return [self.vecs[idx],self.labels[idx]]

class BiGramsTestDataSet(Dataset):
    """ The BiGrams testing dataset containing sentence vectors and actual labels. """
    def __init__(self, datafile):
        self.vecs, self.labels = make_test_vectors(datafile)
    
    def __len__(self):
        return len(self.vecs)

    def __getitem__(self, idx):
        return [self.vecs[idx],self.labels[idx]]


def train(trainloader, model, criterion, optimizer):
    """ Trains a model on a trainset. """
    best_model = model
    best_so_far = -1
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
        
        acc = eval(testloader,net,'data/bag-of-words/outs.tsv','data/bag-of-words/preds.tsv',testset.labels)
        if acc > best_so_far:
            best_model = copy.deepcopy(model)
            best_so_far = acc
    
    print('Finished Training')
    return best_model


def eval(testloader, net, outfile, labelsfile, actual_labels):
    """ 
    Evaluates the results of trained model on a testset and writes results to files:
        1. outfile: Contains all the output softmax tensors (for analysis)
        2. labelsfile: Contains all predicted labels (for analysis)
        3. data/bag-of-words/results.txt: Contains the accuracy on the testset.
    """
    with torch.no_grad():
        outs = []
        preds = []
        for data in testloader:
            # get the vectors; data is a list of [vectors, labels]
            vectors, _ = data
            outputs = net(vectors)
            # get the predicted labels
            _, predicted = torch.max(outputs.data, 1)
            # Add data to lists
            outs.append(outputs)
            preds.append(predicted)

    # WRITE PREDICTED LABELS TO LABELSFILE
        with open(labelsfile,'w') as label_writer:
            label_writer.write("index\tlabel\n")
            # Each batch is of the shape: (BATCH_SIZE * 1)
            # Ex: BATCH_SIZE = 4
            #   [[x],
            #   [x],
            #   [x],
            #   [x]]
    
            # Keep track of no. of sentences
            counter = 0
            for batch in range(len(preds)):
                for idx in range(BATCH_SIZE):
                    label_writer.write("%s\t%s\n" % (counter+1,preds[batch][idx].item()))
                    counter += 1
    
    # WRITE OUTPUT SOFTMAX VECTORS TO OUTFILE    
        with open(outfile,'w') as out_writer:
            out_writer.write("index\ttensor\n")
            # Each batch is of the shape: (BATCH_SIZE * 3)
            # Ex: BATCH_SIZE = 4
            #   [[x,x,x],
            #   [x,x,x],
            #   [x,x,x],
            #   [x,x,x]]

            # Keep track of no. of sentences
            counter = 0
            for batch in range(len(outs)):
                for idx in range(BATCH_SIZE):
                    out_writer.write("%s\t%s\n" % (counter+1,outs[batch][idx]))
                    counter += 1

    # COMPUTE AND WRITE ACCURACY TO RESULTS.TXT
        preds = get_labels(labelsfile)
        accur = simple_accuracy(preds,actual_labels)
        with open('data/count-words/results.txt','w') as results:
            results.write('acc = %s' % str(accur))

        print("Accuracy on dev set: %s" % str(accur))

    print("Finished Testing")
    return accur

if __name__ == "__main__": #### MAIN
    # Set up training
    trainset = BiGramsTrainDataSet('data/bag-of-words/train.tsv')
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

    # Set up testing
    testset = BiGramsTestDataSet('data/bag-of-words/dev.tsv')
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=2)

    # For 2 layer feedforward
    net = two_layer_feedforward(INPUT_SIZE, H1)

    # For 4 layer feedforward
    #net = four_layer_feedforward(INPUT_SIZE, H1, H2, H3)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("Starting Training\n")
    # TRAIN!
    trained_model = train(trainloader, net, criterion, optimizer)

    print("Starting Testing\n")

    # TEST!
    eval(testloader,trained_model,'data/bag-of-words/outs.tsv','data/bag-of-words/preds.tsv',testset.labels)

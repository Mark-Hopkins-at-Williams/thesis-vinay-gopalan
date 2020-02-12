import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from baseline import get_frequencies, create_vocab, create_vectors, get_labels, two_layer_feedforward, simple_accuracy

FREQ_THRESHOLD = 15
BATCH_SIZE = 4
VOCAB = {}
INPUT_SIZE = 0
HIDDEN_LAYER = 784
NUM_EPOCHS = 20

def make_train_vectors(input_filename):
    # Create vectors
    frequencies = get_frequencies(input_filename)
    global VOCAB, INPUT_SIZE
    VOCAB = create_vocab(frequencies, FREQ_THRESHOLD)
    vecs = create_vectors(input_filename, VOCAB)

    # Set value for input size
    INPUT_SIZE = len(vecs[0])

    # Get labels from file
    labels = get_labels(input_filename)

    assert(len(labels)==len(vecs))

    return vecs, labels

def make_test_vectors(input_filename):
    # Create vectors
    vecs = create_vectors(input_filename, VOCAB)

    # Get labels from file
    labels = get_labels(input_filename)

    assert(len(labels)==len(vecs))

    return vecs, labels


class BagOfWordsTrainDataSet(Dataset):
    def __init__(self, datafile):
        self.vecs, self.labels = make_train_vectors(datafile)
    
    def __len__(self):
        return len(self.vecs)

    def __getitem__(self, idx):
        return [self.vecs[idx],self.labels[idx]]


class BagOfWordsTestDataSet(Dataset):
    def __init__(self, datafile):
        self.vecs, self.labels = make_test_vectors(datafile)
    
    def __len__(self):
        return len(self.vecs)

    def __getitem__(self, idx):
        return [self.vecs[idx],self.labels[idx]]


def train(trainloader, model, criterion, optimizer):
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

    print('Finished Training')


def eval(testloader, net, outfile, labelsfile, actual_labels):
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
        
        with open(outfile,'w') as out_writer:
            out_writer.write("index\ttensor\n")
            # Each batch is of the shape: (BATCH_SIZE * 3)
            # Ex: BATCH_SIZE = 4
            #   [[x,x,x],
            #   [x,x,x],
            #   [x,x,x],
            #   [x,x,x]]

            counter = 0
            for batch in range(len(outs)):
                for idx in range(BATCH_SIZE):
                    out_writer.write("%s\t%s\n" % (counter+1,outs[batch][idx]))
                    counter += 1

        # Compute accuracy
        preds = get_labels(labelsfile)
        accur = simple_accuracy(preds,actual_labels)

        print("Accuracy on dev set: %s" % str(accur) )

    print("Finished Testing")

if __name__ == "__main__":
    # Set up training
    trainset = BagOfWordsTrainDataSet('data/SST-3/train.tsv')
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

    net = two_layer_feedforward(INPUT_SIZE, HIDDEN_LAYER)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("Starting Training\n")
    # Train
    train(trainloader, net, criterion, optimizer)

    # Set up testing
    testset = BagOfWordsTestDataSet('data/SST-3/dev.tsv')
    testloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=2)

    print("Starting Testing\n")

    # Test
    eval(testloader,net,'outs.tsv','sentence_preds.tsv',testset.labels)

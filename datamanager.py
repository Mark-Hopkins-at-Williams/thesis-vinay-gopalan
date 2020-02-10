import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from baseline import get_frequencies, create_vocab, create_vectors, get_labels, two_layer_feedforward

FREQ_THRESHOLD = 15
INPUT_SIZE = 0
BATCH_SIZE = 50

def make_train_vectors(input_filename, output_filename=None):
    # Create vectors
    frequencies = get_frequencies(input_filename)
    vocabulary = create_vocab(frequencies, FREQ_THRESHOLD)
    vecs = create_vectors(input_filename, vocabulary)

    # Set value for input size
    global INPUT_SIZE
    INPUT_SIZE = len(vecs[0])

    # Get labels from file
    labels = get_labels(input_filename)

    assert(len(labels)==len(vecs))

    return vecs, labels


class BagOfWordsDataSet(Dataset):
    
    def __init__(self, datafile):
        self.vecs, self.labels = make_train_vectors(datafile)
    
    def __len__(self):
        return len(self.vecs)

    def __getitem__(self, idx):
        return [self.vecs[idx],self.labels[idx]]


def train(trainloader, model, criterion, optimizer):

    for epoch in range(2):  # loop over the dataset multiple times
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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


if __name__ == "__main__":
    # Set up training
    trainset = BagOfWordsDataSet('data/SST-3/train.tsv')
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

    net = two_layer_feedforward(INPUT_SIZE, 784)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("Starting Training\n")
    # Train
    train(trainloader, net, criterion, optimizer)

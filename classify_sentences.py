""" The script used for the setup of classifying sentences using bag-of-words model. """
import random
from clean_data import split_file

##########################################################################################

ALPHABET = ['a','b','c','d','e','f','g']
NUM_SENTENCES = 12000
SENTENCE_LENGTH = 6

##########################################################################################


def make_sentences(num_sentences, sentence_length):
    """ 
    Builds num_sentences sentences of length = sentence_length and returns all sentences in the form of a list.
    """
    sentences = []
    for i in range(num_sentences):
        sentence = ""
        for j in range(sentence_length):
            sentence += random.choice(ALPHABET) + " "

        if sentence not in sentences:
            sentences.append(sentence)
        else:
            while(sentence in sentences):
                sentence = ""
                for j in range(sentence_length):
                    sentence += random.choice(ALPHABET) + " "
            sentences.append(sentence)
    
    return sentences

def make_labels(sentences):
    """ 
    Creates labels for a list of sentences and returns all labels in a list.
    Labels are {0,1,2} to signify negative, neutral and positive respectively.
    """
    labels = []
    for i in range(len(sentences)):
        if 'g' in sentences[i]:
            labels.append(2)
        elif 'e' in sentences[i]:
            labels.append(0)
        else:
            labels.append(1)

    return labels

def sentences_to_tsv(train_file, dev_file):
    """
    Splits the data and writes half in each of the train and dev files.
    """ 
    # Make sentences and labels
    sentences = make_sentences(NUM_SENTENCES,SENTENCE_LENGTH)
    labels = make_labels(sentences)
    assert (len(sentences) == len(labels))
    # Split data
    mid = len(sentences) // 2
    train_sentences, train_labels = sentences[:mid], labels[:mid]
    dev_sentences, dev_labels = sentences[mid:], labels[mid:]

    # Write data to files
    with open(train_file,'w') as train_writer:
        train_writer.write("sentence\tlabel\n")
        for i in range(mid):
            train_writer.write("%s\t%s\n"%(train_sentences[i],train_labels[i]))

    with open(dev_file,'w') as dev_writer:
        dev_writer.write("sentence\tlabel\n")
        for i in range(mid):
            dev_writer.write("%s\t%s\n"%(dev_sentences[i],dev_labels[i]))


if __name__ == "__main__":
    sentences_to_tsv('data/bag-of-words/train.tsv','data/bag-of-words/dev.tsv')
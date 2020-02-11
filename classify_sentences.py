import random
from clean_data import split_file

ALPHABET = ['a','b','c','d','e','f','g']
NUM_SENTENCES = 12000
SENTENCE_LENGTH = 6

def make_sentences(num_sentences, sentence_length):
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
    labels = []
    for i in range(len(sentences)):
        if 'g' in sentences[i]:
            labels.append(2)
        elif 'e' in sentences[i]:
            labels.append(0)
        else:
            labels.append(1)

    return labels

def sentences_to_tsv(filename):
    with open(filename,'w') as writer:
        # Write heading for file
        writer.write("sentence\tlabel\n")
        # Make sentences and labels
        sentences = make_sentences(NUM_SENTENCES,SENTENCE_LENGTH)
        labels = make_labels(sentences)

        assert (len(sentences) == len(labels))

        # Write data to file
        for i in range(len(sentences)):
            writer.write("%s\t%s\n"%(sentences[i],labels[i]))
        
        # Split file into train.tsv and dev.tsv
        split_file('sentence_data.tsv','data/bag-of-words/train.tsv','data/bag-of-words/dev.tsv',0.7,dev_split=True)


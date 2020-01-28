import json
import random

class ConllToken:
    """ Parent token class. """
    def __init__(self, token_type):
        self.token_type = token_type

class Sentiment(ConllToken):
    def __init__(self, sentiment):
        super().__init__("sentiment")
        self.sentiment = sentiment
    
    @staticmethod
    def is_instance(token):
        return token.token_type == "sentiment"

class EndOfSegment(ConllToken):
    """ End of Segment token class. """
    def __init__(self):
        super().__init__("end")
        
    @staticmethod
    def is_instance(token):
        return token.token_type == "end"
        
    def __eq__(self, other):
        return EndOfSegment.is_instance(other)


class Sentiment(ConllToken):
    """ Sentiment token class. """
    def __init__(self, sentiment):
        super().__init__("sentiment")
        self.sentiment = sentiment
    
    @staticmethod
    def is_instance(token):
        return token.token_type == "sentiment"

    def __eq__(self, other):
        return (Sentiment.is_instance(other) and 
                other.sentiment == self.sentiment)


class BasicToken(ConllToken):
    """ Basic token class. """
    def __init__(self, value):
        super().__init__("basic")
        self.value = value

    @staticmethod
    def is_instance(token):
        return token.token_type == "basic"
    
    def __eq__(self, other):
        return (BasicToken.is_instance(other) and 
                other.value == self.value)


class URL(ConllToken):
    """ URL token class. """
    def __init__(self, value):
        super().__init__("url")
        self.value = value

    @staticmethod
    def is_instance(token):
        return token.token_type == "url"
    
    def __eq__(self, other):
        return URL.is_instance(other) and other.value == self.value


class Username(ConllToken):
    """ Username token class. """
    def __init__(self, value):
        super().__init__("username")
        self.value = value

    @staticmethod
    def is_instance(token):
        return token.token_type == "username"
    
    def __eq__(self, other):
        return Username.is_instance(other) and other.value == self.value


def tokenize_conll(lines):
    """ Tokenize lines in a file. """
    for line in lines:
        if line.strip() == "":
            yield EndOfSegment()
        else:
            fields = line.split("\t")
            if fields[0] == "meta":
                yield Sentiment(fields[2].strip())
            else:
                yield BasicToken(fields[0].strip())


def cluster_urls(tokens):
    """ 
    Generator for piecing together URLs in token streams.

    Reads from an instream, and if the instream has broken URL pieces
    (identified by finding 'http') in conll format, the generator yields 
    a complete URL.

    """
    url_builder = ""
    for token in tokens:
        if BasicToken.is_instance(token):
            if url_builder == "" and not token.value.startswith("http"):
                yield token
            elif token.value.startswith("http"):                
                url_builder = token.value
            else:
                url_builder += token.value
        else:
            if url_builder != "":
                yield URL(url_builder)
                url_builder = ""
            yield token
    if url_builder != "":
        yield URL(url_builder) 


def cluster_usernames(tokens):
    """ 
    Generator for piecing together Twitter usernames in token streams.

    Reads from an instream, and if the instream has broken username pieces
    (identified by an '@' symbol) in conll format, the generator 
    yields a complete username.

    """
    builder = ""
    for token in tokens:
        if BasicToken.is_instance(token) and builder == "":
            if token.value == "@":
                builder = token.value
            else:
                yield token
        elif BasicToken.is_instance(token):            
            if builder == "@" or builder[-1] == "_":
                builder += token.value
            elif token.value == "_":
                builder += token.value
            else:
                yield Username(builder)
                builder = ""
                if token.value == "@":
                    builder = token.value
                else:
                    yield token
        else:
            if builder != "":
                yield Username(builder)
                builder = ""
            yield token
    if builder != "":
        yield Username(builder) 


def conll_to_json(conll_file, json_file):
    """ Convert conll format to JSON. """
    result = []
    with open(conll_file) as reader:
        tokens = tokenize_conll([line for line in reader])
        tokens = cluster_urls(tokens)
        tokens = cluster_usernames(tokens)
        next_segment = dict()
        segment_tokens = []
        for tok in tokens:
            if Sentiment.is_instance(tok):
                next_segment['sentiment'] = tok.sentiment
            elif EndOfSegment.is_instance(tok):
                next_segment['segment'] = ' '.join(segment_tokens)
                result.append(next_segment)
                next_segment = dict()
                segment_tokens = []
            elif BasicToken.is_instance(tok):
                segment_tokens.append(tok.value)
    if 'sentiment' in next_segment:
        next_segment['segment'] = ' '.join(segment_tokens)
        result.append(next_segment)
    with open(json_file, 'w') as writer:
        writer.write(json.dumps(result, indent=4))


def conll_to_tsv(conll_file, tsv_file):
    """ Convert conll format to TSV. """
    with open(conll_file) as reader:
        with open(tsv_file, 'w') as writer:
            tokens = tokenize_conll([line for line in reader])
            tokens = cluster_urls(tokens)
            tokens = cluster_usernames(tokens)
            writer.write("sentence\tlabel\n")
            segment_tokens = []
            sentiment = -1
            for tok in tokens:
                if Sentiment.is_instance(tok):
                    if tok.sentiment == "negative":
                        sentiment = 0
                    elif tok.sentiment == "neutral":
                        sentiment = 1
                    elif tok.sentiment == "positive":
                        sentiment = 2
                elif EndOfSegment.is_instance(tok):                    
                    next_segment = ' '.join(segment_tokens)
                    next_segment += '\t' + str(sentiment)
                    if sentiment >= 0:
                        writer.write(next_segment + '\n')
                    sentiment = -1
                    segment_tokens = []
                elif BasicToken.is_instance(tok):
                    segment_tokens.append(tok.value)
            if sentiment >= 0:
                next_segment = ' '.join(segment_tokens)
                next_segment += '\t' + str(sentiment)
                writer.write(next_segment + '\n')


def split_file(input_file, out1, out2, percentage, dev_split=False):
    """ 
    Splits an input file into two output files based on the percentage.
    May be used to generate both test and dev files.
    """
    with open(input_file, 'r') as reader:
        with open(out1, 'w') as writer75:
            with open(out2, 'w') as writer25:
                if not dev_split:
                    writer25.write("index\tsentence\n")
                else:
                    writer25.write("sentence\tlabel\n")
                lines_in_smaller = 0
                for line in reader:
                    r = random.random()
                    if r < percentage:
                        writer75.write(line)
                    else:
                        if not dev_split:
                            line = line.split("\t")[0]
                            writer25.write(str(lines_in_smaller) + "\t" + line + "\n")
                            lines_in_smaller += 1
                        else:
                            writer25.write(line)

def testify(input_file, output_file):
    """ Formats given trial data into test data. """
    with open(input_file, 'r') as reader:
        with open(output_file, 'w') as writer:
            writer.write("index\tsentence\n")
            index = 0
            for line in reader:
                line = line.split("\t")[0]
                if line != "sentence":
                    writer.write(f"{index}\t{line}\n")
                    index += 1


if __name__ == "__main__":
    conll_to_tsv('data/train_14k_split_conll.txt','data/SST-3/train.tsv')
    conll_to_tsv('data/dev_3k_split_conll.txt','data/SST-3/dev.tsv')


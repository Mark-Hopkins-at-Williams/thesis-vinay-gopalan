import json

class ConllToken:
    def __init__(self, token_type):
        self.token_type = token_type
 
class EndOfSegment(ConllToken):
    def __init__(self):
        super().__init__("end")
        
    @staticmethod
    def is_instance(token):
        return token.token_type == "end"
        
    def __eq__(self, other):
        return EndOfSegment.is_instance(other)
    

class Sentiment(ConllToken):
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
    def __init__(self, value):
        super().__init__("url")
        self.value = value

    @staticmethod
    def is_instance(token):
        return token.token_type == "url"
    
    def __eq__(self, other):
        return URL.is_instance(other) and other.value == self.value
  
class Username(ConllToken):
    def __init__(self, value):
        super().__init__("username")
        self.value = value

    @staticmethod
    def is_instance(token):
        return token.token_type == "username"
    
    def __eq__(self, other):
        return Username.is_instance(other) and other.value == self.value

     
    
def tokenize_conll(lines):
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
            
            
   
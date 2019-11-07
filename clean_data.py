


def cluster_urls(instream):
    """ 
    Generator for piecing together URL streams.

    Reads from an instream, and if the instream has broken URL pieces
    (identified by finding 'http') in conll format, the generator yields a complete URL.

    """
    complete = False
    next_token = ""
    url_flag = False
    while not complete:
        try:
            line = next(instream)
            fields = line.split('\t')
            first_word = fields[0].strip()
            next_token += first_word
            if not url_flag and not first_word.startswith("http"):
                yield next_token 
                next_token = ""
            elif first_word.startswith("http"):
                url_flag = True
            elif url_flag and first_word == "":
                url_flag = False
                yield next_token
                next_token = ""
                yield "**DONE**"
            if first_word == "meta":
                yield fields[2].strip()
        except StopIteration:
            complete = True


def cluster_users(instream):
    """ 
    Generator for piecing together Username streams.

    Reads from an instream, and if the instream has broken username pieces
    (identified by an '@' symbol) in conll format, the generator yields a complete URL.

    """
    complete = False
    next_token = ""
    user_flag = False
    is_next_score = True
    while not complete:
        try:
            line = next(instream)
            fields = line.split('\t')
            first_word = fields[0].strip()
            if not user_flag and first_word != "@" and first_word != "_":
                next_token += first_word
                yield next_token 
                next_token = ""
            elif first_word == "@" or first_word == "_":
                next_token += first_word
                user_flag = True
                is_next_score = True
            elif user_flag and first_word != "@" and first_word != "_":
                if is_next_score:
                    next_token += first_word
                    is_next_score = False
                else:
                    user_flag = False
                    yield next_token
                    next_token = ""
                    next_token += first_word
                    yield next_token
                    next_token = ""
            if first_word == "meta":
                yield fields[2].strip()
        except StopIteration:
            complete = True

def process(): 
    """ 
    Process that writes two new txt files: clean_train and final_train

    clean_train.txt is the file written with complete URLs but incomplete usernames.
    (It is used to create final_train.txt)

    final_train.txt is the file written with complete URLs and complete usernames.
    (It is the data to be used) 

    """
    with open('data/train_conll.txt','r') as data:
        with open('data/clean_train.txt','w') as clean_data:
            url_flag = 0
            url_items = []
            for line in data:
                if url_flag == 0 and "https" not in line:
                    clean_data.write(line)
                if "https" in line:
                    url_flag = 1
                if url_flag == 1 and line != "\n":
                    url_items.append(line)
                if line == "\n" and url_flag == 1:
                    url = [x.replace('\t','').replace('\n','') for x in url_items]
                    for i in range(0,len(url)):
                        if 'Eng' in url[i]:
                            item = url[i].replace('Eng','')
                            url[i] = item
                        if 'Hin' in url[i]:
                            item = url[i].replace('Hin','')
                            url[i] = item
                        if 'O' in url[i]:
                            item = url[i].replace('O','')
                            url[i] = item
                    url_string = ''.join(url)
                    clean_data.write(url_string + "\t" + "O")
                    clean_data.write("\n")
                    clean_data.write("\n")
                    url_flag = 0
                    url_items = []
    with open('data/clean_train.txt','r') as url_clean_data:
        with open('data/final_train.txt','w') as final_data:
            user_flag = 0
            user_items = []
            for line in url_clean_data:
                if "@" in line or "_" in line:
                    user_items.append(line)
                    user_flag = 1
                if user_flag == 0:
                    if len(user_items) > 0:
                        user = [x.replace('\t','').replace('\n','') for x in user_items]
                        for i in range(0,len(user)):
                            if 'Eng' in user[i]:
                                item = user[i].replace('Eng','')
                                user[i] = item
                            if 'Hin' in user[i]:
                                item = user[i].replace('Hin','')
                                user[i] = item
                            if 'O' in user[i]:
                                item = user[i].replace('O','')
                                user[i] = item
                        user_string = ''.join(user)
                        final_data.write(user_string + "\t" + "O")
                        final_data.write("\n")
                        user_items = []
                    if "@" not in line and "_" not in line:
                        final_data.write(line)
                    if user_flag == 1 and "@" not in line and "_" not in line:
                        user_items.append(line)
                        user_flag = 0
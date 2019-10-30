""" Script for cleanup of training data. """

with open('train_conll.txt','r') as data:
    with open('clean_train.txt','w') as clean_data:
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


with open('clean_train.txt','r') as url_clean_data:
    with open('final_train.txt','w') as final_data:
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
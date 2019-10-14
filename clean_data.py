""" Script for cleanup of training data. """

with open('train_conll.txt','r') as data:
    with open('clean_train.txt','w') as clean_data:
        url_flag = 0
        url_items = []
        for line in data:
            if url_flag == 0:
                clean_data.write(line)
            if "…" in line:
                url_flag = 1
            if url_flag == 1 and "…" not in line and line != "\n":
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
from clean_data import conll_to_json

def format_results(json_file, preds_file, outfile):
    tweets_data = []
    preds = []
    with open(json_file,'r') as json_reader:
        with open(preds_file,'r') as preds_reader:
            for tweet in json_reader:
                tweets_data.append(tweet)
            for label in preds_reader:
                preds.append(label)
    
    assert (len(tweets_data) == len(preds))
    num_lines = len(tweets_data)
    with open(outfile,'w') as writer:
        writer.write('Uid,Sentiment\n')
        for i in range(num_lines-1):
            writer.write('%s,%s\n'%(tweets_data[i]['uid'],preds[i]))
        # To ensure no blank line at EOF
        writer.write('%s,%s'%(tweets_data[num_lines-1]['uid'],preds[num_lines-1]))

if __name__ == "__main__":
    conll_to_json('data/dev_3k_split_conll.txt','data/SST-3/dev_data.json')
    format_results('data/SST-3/dev_data.json','data/SST-3/checks/labels.txt','data/SST-3/answer.txt')


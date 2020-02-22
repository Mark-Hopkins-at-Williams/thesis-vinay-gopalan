from clean_data import conll_to_json
import json

def format_results(json_file, preds_file, outfile):
    preds = []
    with open(json_file,'r') as json_reader:
        data = json.load(json_reader)
    with open(preds_file,'r') as preds_reader:
        for label in preds_reader:
            preds.append(label)
    
    assert (len(data) == len(preds))
    num_lines = len(data)
    with open(outfile,'w') as writer:
        writer.write('Uid,Sentiment\n')
        for i in range(num_lines-1):
            if int(preds[i]) == 0:
                writer.write('%s,negative\n'%(data[i]['uid']))
            elif int(preds[i]) == 1:
                writer.write('%s,neutral\n'%(data[i]['uid']))
            else:
                writer.write('%s,positive\n'%(data[i]['uid']))

        # To ensure no blank line at EOF
        if int(preds[num_lines-1]) == 0:
            writer.write('%s,negative'%(data[num_lines-1]['uid']))
        elif int(preds[num_lines-1]) == 1:
            writer.write('%s,neutral'%(data[num_lines-1]['uid']))
        else:
            writer.write('%s,positive'%(data[num_lines-1]['uid']))


if __name__ == "__main__":
    conll_to_json('data/Hindi_test.txt','data/SST-3/dev_data.json')
    format_results('data/SST-3/dev_data.json','data/SST-3/checks/labels.txt','data/SST-3/answer.txt')


""" Script to append glue training data to SST-3 training data. """

def append_glue_data(data_file, glue_file):
    with open(data_file,'a') as writer:
        with open(glue_file,'r') as reader:
            for line in reader:
                line = line.split("\t")
                line[1] = line[1].strip()
                if line[0] != 'sentence':
                    if line[1] == '0':
                        writer.write(f"{line[0]}\t0\n")
                    else:
                        writer.write(f"{line[0]}\t2\n")

if __name__ == "__main__":
    append_glue_data('data/SST-3/train.tsv','data/glue_data/SST-2/train.tsv')
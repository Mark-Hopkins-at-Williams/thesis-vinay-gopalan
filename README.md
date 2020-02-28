
# Sentiment Analysis on Code-Mixed Tweets

### Collated Results

The evaluation results of all experiments can be found at https://docs.google.com/spreadsheets/d/1uTEbuPMO01eKuVeVVugxpmxJLWidkFj3TWun7URgqQs/edit?usp=sharing

## Fine-tuning Model

### Getting the data and setting it up

1. In the repository root, create a directory `data`.

2. Download the training data `train_14k_split_conll.txt` from Google Drive at https://drive.google.com/file/d/183M82UWxzFS6cYJQ76_gExfqSLo3T3Mb/view?usp=sharing and extract it to `data`.

3. Downlaod the validation data `dev_3k_split_conll.txt` from Google Drive at https://drive.google.com/file/d/1lhn9nyrwVFz5tJiw1J_ThswOKyyaNXmk/view?usp=sharing and extract it to `data`.

4. Download the test data from Google Drive at https://drive.google.com/file/d/1WTOjLs-R6HhWT8skEeit4jcsEgjxeOeE/view and extract it  to `data` with the name `Hindi_test.txt`.

5. Inside `data`, create a directory `SST-3`.

6. Inside `SST-3`, create a directory `checks`.

7. From the root directory, run `python3 clean_data.py`.

8. (If you wish to append SST-2 data) From the root directory, run `python3 append_glue.py`.

### Run GLUE experiments

From the root directory, run `sh glue.sh`.

### Results

The results of the experiements on the dev set will be in the text file `eval_results.txt` in the specified output_dir `data/SST-3/checks/`.

## Bag-of-words Model

### Setup

1. Inside `data`, create a directory `bag-of-words`.

### Experiments

#### Classify SST-3 tweets dataset

NOTE: Before step 2, make sure that the you copy `dev.tsv` and `train.tsv` to `data/bag-of-words/` folder.

1. From the root directory, run `python3 clean_data.py` to create tweet data. You will find train.tsv and dev.tsv in the folder `data/SST-3`.

2. Copy both files to `data/bag-of-words`.

3. From the root directory, run `python3 bag_of_words.py`.

#### Use BiGrams

NOTE: Before step 2, make sure that the you copy `dev.tsv` and `train.tsv` to `data/bag-of-words/` folder.

1. From the root directory, run `python3 clean_data.py` to create tweet data. You will find train.tsv and dev.tsv in the folder `data/SST-3`.

2. Copy both files to `data/bag-of-words`.

3. From the root directory, run `python3 bow_bigrams.py`.

### Results

The accuracy results of the experiements on the dev set will be in the file `data/bag-of-words/results.txt`. Furthermore, all softmax output vectors are in `data/bag-of-words/outs.tsv` in the root directory and all predicted labels are in `data/bag-of-words/preds.tsv`. 

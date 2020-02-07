

# Sentiment Analysis on Code-Mixed Tweets


### Collated Data
The evaluation results of all experiments can be found at https://docs.google.com/spreadsheets/d/1uTEbuPMO01eKuVeVVugxpmxJLWidkFj3TWun7URgqQs/edit?usp=sharing

### Getting the data and setting it up

1. In the repository root, create a directory `data`.

2. Download the training data `train_14k_split_conll.txt` from Google Drive at https://drive.google.com/file/d/183M82UWxzFS6cYJQ76_gExfqSLo3T3Mb/view?usp=sharing and extract it to `data`.

3. Downlaod the validation data `dev_3k_split_conll.txt` from Google Drive at https://drive.google.com/file/d/1lhn9nyrwVFz5tJiw1J_ThswOKyyaNXmk/view?usp=sharing and extract it to `data`.

4. Inside `data`, create a directory `SST-3`.

5. Inside `SST-3`, create a directory `checks`.

6. From the root directory, run `python3 clean_data.py`.

7. (If you wish to append SST-2 data) From the root directory, run `python3 append_glue.py`.

### Run GLUE experiments

1. From the root directory, run `sh glue.sh`.

2. You can find the results of the experiments in the directory `data/SST-3/checks` in the file `eval_results.txt`.

### Run Baseline experiments

1. From the root directory, run `python3 baseline.py`.

2. You can find the results in `base.txt` in the root directory.

### Results

1. The results of the experiements on the dev set will be in the text file `eval_results.txt` in the specified output_dir `data/SST-3/checks/`.

2. The results of the baseline experiments will be in the text file `base.txt` in the root directory.

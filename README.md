

# Sentiment Analysis on Code-Mixed Tweets


### Getting the data and setting it up

1. In the repository root, create a directory `data`.

2. Download `train_conll.txt` from Google Drive at https://drive.google.com/file/d/1fzhNn_baYWE6Ltxf1VKlkYzCUwUlHVXx/view?usp=sharing/ and extract it to `data`.

3. Downlaod `trial_conll.txt` from Google Drive at https://drive.google.com/file/d/1T_AvUYwc8Fld_8VoUj9s9S1ZbmpkAsVI/view?usp=sharing/ and extract it to `data`.

4. Inside `data`, create a directory `SST-3`.

5. Inside `SST-3`, create a directory `checks`.

6. From root, run `python3 clean_data.py`.

7. (If you wish to append SST-2 data) From root, run `python3 append_glue.py`.

### Run GLUE experiments

From root, run `sh glue.sh`.

### Results

The dev set results will be present within the text file 'eval_results.txt' in the specified output_dir `data/SST-3/checks/`

#### TWITTER CORPORA

1. A corpus of 476 million tweets (SNAP):
http://www.google.com/url?q=http%3A%2F%2Fsnap.stanford.edu%2Fdata%2Fbigdata%2Ftwitter7%2Ftweets2009-06.txt.gz&sa=D&sntz=1&usg=AFQjCNGwmQrrykdp2jPXKmQwfbxbeTg8EQ


2. A corpus of 1.6 million tweets (Sentiment140):
https://docs.google.com/file/d/0B04GJPshIjmPRnZManQwWEdTZjg/edit


3. A corpus of 4242 tweets (Sentiment Strength):
http://sentistrength.wlv.ac.uk/documentation/

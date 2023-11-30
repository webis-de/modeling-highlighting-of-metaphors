import pandas as pd
from sklearn.model_selection import train_test_split
import os

from project_metaphor import PROJECT_ROOT
DATASETS_PATH = os.path.join(PROJECT_ROOT,'datasets/')

# '''
# Make the dataset file with positive and negative examples 
# '''
# df = pd.read_csv(DATASETS_PATH+'metaphor-corpus.csv')
# df_neg = pd.read_csv(DATASETS_PATH+'negative-source-cm.csv')
# df.insert(10,"Negative Source CM",df_neg['Source CM'])
# df.to_csv(DATASETS_PATH+"all-pos-neg-scm.csv")

# '''
# This dataset split needs to be in a correct way. Things to note:
# 1. Every positive ground truth should stay at least once in the train set 
# ... more to add maybe ... 
# '''

'''
split to train-dev-test
'''

df = pd.read_csv(DATASETS_PATH+'metaphor-corpus.csv')
df = df.drop_duplicates(subset=['Sentence'])
train, rest = train_test_split(df, test_size=0.3)
dev, test = train_test_split(rest, test_size=0.7)

train.to_csv(DATASETS_PATH+'metaphor-scm-train-part1.csv')
dev.to_csv(DATASETS_PATH+'metaphor-scm-dev-part1.csv')
test.to_csv(DATASETS_PATH+'metaphor-scm-test-part1.csv')
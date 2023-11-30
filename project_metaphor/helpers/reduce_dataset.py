import pandas as pd
from sklearn.model_selection import train_test_split
import os

from project_metaphor import PROJECT_ROOT
DATASETS_PATH = os.path.join(PROJECT_ROOT,'datasets/')

''' function to reduce dataset to only features we need'''

def dataset_reducer(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df[['Sentence','Schema Slot']]
    df.to_csv(DATASETS_PATH+'metaphor-hghl-basic-test.csv', index=False)

train_path = DATASETS_PATH+'metaphor-train.csv'
dev_path = DATASETS_PATH+'metaphor-dev.csv'
test_path = DATASETS_PATH+'metaphor-test.csv'

dataset_reducer(test_path)

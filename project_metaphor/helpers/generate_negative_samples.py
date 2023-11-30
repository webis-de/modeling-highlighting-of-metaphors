import pandas as pd
import csv
import random
import os

from project_metaphor import PROJECT_ROOT
DATASETS_PATH = os.path.join(PROJECT_ROOT,'datasets/')

# def rephrase_list(x,ls):
#     FINAL = list()
#     for each in ls:
#         if each!=x:
#             FINAL.append(each)
#         else:
#             continue
#     return FINAL

# def check(stuff,unq):
#     unq1=rephrase_list(stuff,unq)
#     val = random.choice(unq1)
#     return val 
    
# '''
# Get data
# '''
# df = pd.read_csv(DATASETS_PATH+'metaphor-corpus.csv')
# df['Source Status'] = 0
# unq = df['Source CM'].unique().tolist()
# df['Source CM'] = df.apply(lambda x: check(x['Source CM'],unq),  axis=1)

# '''
# Make temporary file for negative samples 
# '''

# df.to_csv(DATASETS_PATH+'negative-source-cm.csv',index=False)

# '''
# Merge positive and negative samples 
# '''

# df1 = pd.read_csv(DATASETS_PATH+'metaphor-corpus.csv')
# df1['Source Status'] = 1
# df2 = pd.read_csv(DATASETS_PATH+'negative-source-cm.csv')
# df3 = pd.concat([df1,df2])

# '''
# Shuffle the newly made dataset/dataframe randomnly
# '''

# df3 = df3.sample(frac=1).reset_index(drop=True)
# df3.drop('Unnamed: 11', inplace=True, axis=1)
# df3.drop('Unnamed: 12', inplace=True, axis=1)

# df3.to_csv(DATASETS_PATH+'all-positive-negative-source-cm.csv')

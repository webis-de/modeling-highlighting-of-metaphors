import pandas as pd

TRAIN_PATH = '../datasets/metaphors-train-part1-clean-csr.csv'
DEV_PATH = '../datasets/metaphors-dev-part1-clean-csr.csv'  
TEST_PATH = '../datasets/metaphors-test-part1-clean-csr.csv'

tr = pd.read_csv(TRAIN_PATH)
dv = pd.read_csv(DEV_PATH)
ts = pd.read_csv(TEST_PATH)

sent_tr = list(tr['Sentence'])
sent_dv = list(dv['Sentence'])
sent_ts = list(ts['Sentence'])

for each in sent_tr:
    if each in sent_ts:
        print('YES')
        break
    else:
        print('No')

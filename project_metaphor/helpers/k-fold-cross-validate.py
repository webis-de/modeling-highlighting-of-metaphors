import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# df = pd.read_csv('../datasets/metaphor-corpus.csv')

# X = df
# y = df['Target LM']

# X.drop_duplicates(subset=['Sentence'])

# kf = KFold(n_splits = 3, shuffle = True, random_state = 2)

# i=1
# for train_index, test_index in kf.split(X):
#     X_tr_va, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_tr_va, y_test = y[train_index], y[test_index]
#     X_train, X_val, y_train, y_val = train_test_split(X_tr_va, y_tr_va, test_size=0.2, random_state=1)
#     X_train.to_csv('../datasets/3-fold-cross-validation-train-{}.csv'.format(i))  
#     X_val.to_csv('../datasets/3-fold-cross-validation-val-{}.csv'.format(i)) 
#     X_test.to_csv('../datasets/3-fold-cross-validation-test-{}.csv'.format(i)) 
#     i+=1


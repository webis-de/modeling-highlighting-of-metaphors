import pandas as pd

df_train = pd.read_csv('metaphors-train-part1-clean-csr.csv')  
df_dev = pd.read_csv('metaphors-dev-part1-clean-csr.csv')
df_test = pd.read_csv('metaphors-test-part1-clean-csr.csv')

all_dfs = [df_train,df_dev,df_test]

df = pd.concat(all_dfs).reset_index(drop=True)

hghl_all = list(df['Schema Slot'].unique())

dict_count = dict()
for key in df['Schema Slot'].to_list():
    if key not in dict_count.keys():
        dict_count[key] = 1
    else:
        dict_count[key] += 1

def avg(__dict):
    return sum(__dict.values()) / len(__dict)

avg_hghl = avg(dict_count)

less_than_ten = list()
more_than_ten = list()

for key,val in dict_count.items():
        if val <= 10:
            less_than_ten.append(key)
        elif val > 10:
            more_than_ten.append(key) 
'''
Check thrice this step
'''

less = list()
more = list()

for i in range(len(df_test['Schema Slot'])):
    hghl = df_test['Schema Slot'][i]
    sentence = df_test['Sentence'][i]
    source_cm = df_test['Source CM'][i]
    source_lm = df_test['Source LM'][i]

    if hghl in less_than_ten:
        less.append([sentence,hghl,source_cm,source_lm])
    else:
        more.append([sentence,hghl,source_cm,source_lm])

df_less = pd.DataFrame(less, columns = ['Sentence', 'Schema Slot','Source CM','Source LM'])
df_more = pd.DataFrame(more, columns = ['Sentence', 'Schema Slot','Source CM','Source LM'])

df_less.to_csv('hghl-test-less-than-ten.csv',index=False)
df_more.to_csv('hghl-test-more-than-ten.csv',index=False)

 
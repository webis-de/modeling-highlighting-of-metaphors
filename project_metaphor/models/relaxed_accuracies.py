import pandas as pd

'''
Highlighted Aspects
'''

'''
No Pretrian Single
'''
# RESULTS_PATH = '../analysis/single-predict-hghl-nopretrain-relaxed-acc-nometaphor.csv'
'''
With Pretrian Single
'''
# RESULTS_PATH = '../analysis/single-predict-hghl-withpretrain-relaxed-acc-nometaphor.csv'
'''
Multi 
'''
# RESULTS_PATH = '../analysis/multi-predict-hghl-nopretrain-relaxed-acc-nometaphor.csv'

'''
Source Domains
'''

'''
No Pretrian Single
'''
# RESULTS_PATH = '../analysis/single-predict-scm-nopretrain-relaxed-acc-nometaphor.csv'

'''
With Pretrian Single
'''
RESULTS_PATH = '../analysis/single-predict-scm-withpretrain-relaxed-acc-nometaphor.csv'

'''
Multi 
'''
# RESULTS_PATH = '..//'

def acc_conditions(true,pred):
    # print(true)
    # print(pred)
    if true == pred:
        return 'correct'
    
    elif '/' in true:
        true = true.split('/')
        if pred in true: 
            return 'correct'
    
    elif '/' in pred:
        pred = pred.split('/')
        if true in pred: 
            return 'correct'
    
    elif '/' in pred and '/' in true:
        true = true.split('/')
        pred = pred.split('/')
        for each in true:
            if each in pred:
                break
                return 'correct'
    else:
        return 'Incorrect'
    
def measure_acc(true_list,pred_list):
    assert len(true_list) == len(pred_list)
    total = len(true_list)
    count = 0
    for i,j in zip(true_list,pred_list):
        if acc_conditions(i,j) == 'correct':     
            count+=1
    
    return count/total


df = pd.read_csv(RESULTS_PATH)

__true = df['True'].to_list()
__pred = df['Prediction'].to_list()

print(measure_acc(__true,__pred))

'''
Approach is a sexy one
Multi task setup captures overall real world situation much better
'''
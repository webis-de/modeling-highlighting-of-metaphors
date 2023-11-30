from scipy import stats
#from statsmodels.stats import weightstats 
from scipy.stats import ttest_rel
import pandas as pd

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df_x = pd.read_csv('../analysis/results-scm-multi-predict-scm-nopretrain-sigtest.csv')
df_y = pd.read_csv('../analysis/results-scm-single-predict-scm-nopretrain-sigtest.csv')

true = df_x['True'].to_list()
predicted_x = df_x['Prediction'].to_list()
predicted_y = df_y['Prediction'].to_list()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    final = []
    for i in range(0, len(lst), n):
        temp = lst[i:i + n]
        final.append(temp)

    return final

def chunk_acc(true, chunk_any):
    correct = 0
    assert len(true)==len(chunk_any)
    total = len(chunk_any)
    for i,j in zip(true,chunk_any):
        if i==j:
            correct +=1 

    acc = correct/total

    return acc
    
def get_chunk_accuracies(chunk_true,chunk_x,chunk_y):
    acc_list = list()
    
    assert len(chunk_true)==len(chunk_x)
    assert len(chunk_true)==len(chunk_y)

    final_x = list()
    for i in range(len(chunk_true)): 
        temp_acc1 = chunk_acc(chunk_true[i],chunk_x[i])
        final_x.append(temp_acc1)

    final_y = list()
    for j in range(len(chunk_true)): 
        temp_acc2 = chunk_acc(chunk_true[j],chunk_y[j])
        final_y.append(temp_acc2)

    return (final_x,final_y)

#Check wither v1s are signficantly better than v2s
def check_sig(v1s, v2s, alpha=0.5):

    diff = list(map(lambda x1 , x2: x1 - x2, v1s, v2s))
    is_normal = stats.shapiro(diff)[1] > alpha
    
    if is_normal:
        print('Distribution is normal, so using ttest_rel')
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
        result = ttest_rel(v1s, v2s, alternative='greater')
        tstat, pvalue= result.statistic, result.pvalue
        print('tstat : ', tstat, '\npvalue : ', pvalue)
        if tstat >=0:
            if (pvalue) <= alpha:
                return True
            else:
                return False
        else:
            return False

    else:
        print('Distribution is not normal, so using wilcoxon')
        ttest = stats.wilcoxon(v1s, v2s, alternative='greater')
        
        if ttest.statistic >=0:
            print('ttest.pvalue ', ttest.pvalue)
            if (ttest.pvalue) <= alpha:
                return True
            else:
                return False
        else:
            return False
                

chunk_true = chunks(true,50)
chunk_multi = chunks(predicted_x,50)
chunk_single = chunks(predicted_y,50)

x, y = get_chunk_accuracies(chunk_true,chunk_multi,chunk_single)

print(check_sig(x,y))



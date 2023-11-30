import os
import pandas as pd

from sklearn.metrics import accuracy_score
# from project_metaphor import PROJECT_ROOT

# DATASETS_PATH = os.path.join(PROJECT_ROOT,'datasets/')

def zero_rule_algorithm_classification(train, test):
	output_values = [row for row in train['Source CM']]
	prediction = max(set(output_values), key=output_values.count)
	predicted = [prediction for i in range(len(test))]
	return predicted

train = pd.read_csv('metaphor-scm-train.csv')
test = pd.read_csv('metaphor-scm-test.csv')

y_pred = test['Source CM'].tolist()
y_test = zero_rule_algorithm_classification(train,test)

acc = accuracy_score(y_test,y_pred)

## Acc = 0.08602150537634409 for source domains
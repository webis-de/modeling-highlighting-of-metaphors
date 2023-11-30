from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib as plt

df_x = pd.read_csv('../analysis/results-scm-multi-predict-scm-nopretrain-sigtest.csv')
df_y = pd.read_csv('../analysis/results-scm-single-predict-scm-nopretrain-sigtest.csv')

y_test = df_x['True'].to_list()
y_pred = df_x['Prediction'].to_list()
# y_pred = df_y['Prediction'].to_list()

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
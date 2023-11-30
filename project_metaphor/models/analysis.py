from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

scaler = MinMaxScaler()
np.random.seed(42)

'''
    Take the embedding of stuff from before like with distillbert of random samples and then after the pretrained model is done 
'''

ANALYSIS_PATH_TRAIN = '../datasets/metaphors-train-part1-clean-csr.csv'
ANALYSIS_PATH_DEV = '../datasets/metaphors-dev-part1-clean-csr.csv'
ANALYSIS_PATH_TEST = '../datasets/metaphors-test-part1-clean-csr.csv'

df_train = pd.read_csv(ANALYSIS_PATH_TRAIN)
df_dev = pd.read_csv(ANALYSIS_PATH_DEV)
df_test = pd.read_csv(ANALYSIS_PATH_TEST)

df = df_train + df_dev + df_test

model_name = 'distilbert-base-uncased'

best_model_predict_source_domain_single_task_setup = './model_config/predict-scm/distilbert-base-uncased-hghl-single-task-2023-04-17_04-52-51/'
best_model_predict_source_domain_joint_learning_setup = './model_config_multi_task/predict-scm/distilbert-base-uncased-multi-all-scm-2023-04-20_16-08-43/'
best_model_predict_source_domain_continual_learning_setup = './model_config/predict-scm/./model_config/predict-hghl/distilbert-base-uncased-hghl-single-task-2023-04-15_14-22-11/-scm-single-task-2023-04-20_05-40-09/'

best_model_predict_highlighted_aspects_single_task_setup = './model_config/predict-hghl/distilbert-base-uncased-hghl-single-task-2023-04-15_14-22-11/'
best_model_predict_highlighted_aspects_joint_learning_setup = './model_config_multi_task/MultiAll/distilbert-base-uncased-multi-all-hghl-2023-02-25_21-57-17/'
best_model_predict_highlighted_aspects_continual_learning_setup = './model_config/predict-hghl/./model_config/predict-scm/distilbert-base-uncased-hghl-single-task-2023-04-17_04-52-51/-hghl-single-task-2023-05-04_04-19-04/'

max_seq_length = 256
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

hghlaspcts_dict = dict()
srcdmns_dict = dict()
srcmtphrs_dict = dict()

sentences = df['Sentence']
hghlaspcts = df['Schema Slot'] 
srcdmns = df['Source CM']

embedding_before_trainging = model.encode(list(sentences))

'''
This is where Milad's analysis starts
'''


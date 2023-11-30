"""
This script trains sentence transformers with a triplet loss function.
As corpus, we use the wikipedia sections dataset that was describd by Dor et al., 2018, Learning Thematic Similarity Metric Using Triplet Networks.
"""
from ast import Raise
import pandas as pd
import csv
import logging
import os
import sys
# from project_metaphor import PROJECT_ROOT
from datetime import datetime
from zipfile import ZipFile
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
#from sentence_transformers.datasets import SentenceLabelDataset
#from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, losses, models, util
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator
from TripletRankingEvaluator import RankingEvaluator 
import wandb
from transformers import TrainingArguments, Trainer
from transformers import AutoConfig
import gc
import argparse

'''
Argparse
'''

'''
Uncomment for manual sweeps
'''

parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument("itr", type=int)
parser.add_argument("task", type=str)
parser.add_argument("epochs", type=int)
parser.add_argument("train_batch_size", type=int)
parser.add_argument("learning_rate", type=float)
parser.add_argument("add_xIntent", type=int)
parser.add_argument("test_split", type=int)
parser.add_argument("sub_task", type=str)
parser.add_argument("eval_subset", type=str) 
parser.add_argument("specific_output_folder",type=str)
args = parser.parse_args()

'''
Uncomment for manual sweeps
'''
itr = args.itr
task = args.task
epochs = args.epochs
train_batch_size = args.train_batch_size
learning_rate = args.learning_rate
add_xIntent = args.add_xIntent
test_split = args.test_split
sub_task = args.sub_task
eval_subset = args.eval_subset
specific_output_folder = args.specific_output_folder

'''
Uncomment only for sweeps 
'''
# itr = 0
# task = 'single'
# epochs = 'egal'
# train_batch_size = 'egal'
# learning_rate = 'egal'
# add_xIntent = 0
# test_split = 0
# sub_task = 'scm'
# eval_subset = 'dev'
# specific_output_folder = 'scmhypsweeptest'

print('\n', 'Iteration: ', itr, '\n')
print('\n', 'Task: ', task, '\n')
print('\n', 'Epochs: ', epochs, '\n')
print('\n', 'Train Batch Size: ', train_batch_size, '\n')
print('\n', 'Learning Rate: ', learning_rate, '\n')
print('\n', 'Sub Task: ', sub_task, '\n')
print('\n', 'Eval Subset: ', eval_subset, '\n')
print('\n', 'xAll: ', add_xIntent, '\n')
print('\n', 'Specific Output Folder Path: ', specific_output_folder, '\n')
# epochs for hghl = 4

def get_model_config(model_name):
    config = AutoConfig.from_pretrained(model_name)
    return config

# model_name = 'bert-base-uncased' # works
# best to predict highlighted aspects with metaphors without any pretrain
# model_name = './model_config/SingleHghlAll/distilbert-base-uncased-single-hghl-all-2023-03-08_19-10-56/'
# model_name = './model_config/predict-hghl/distilbert-base-uncased-hghl-single-task-2023-04-15_14-22-11/' #best-without-scm-predict-hghl
# model_name = './model_config/predict-scm/distilbert-base-uncased-hghl-single-task-2023-04-17_04-52-51/'
# model_name = 'distilbert-base-uncased' 
# model_name = 'bert-large-uncased' # works with batch size 16 all acc 0.00 - 0.04
# model_name = 'sentence-transformers/all-mpnet-base-v2' # works
# model_name = 'albert-base-v2' # works
# model_name = 'xlm-roberta-base' # all acc 0.00
# model_name = 'roberta-large' # works 
# model_name = 'albert-base-v2'

'''best checkpoints deberta'''

# model_name = 'microsoft/deberta-base'
model_name = './model_config/predict-scm/microsoft/deberta-base-scm-single-setup-2023-05-26_18-25-37/'

# model_name = './model_config/predict-hghl/microsoft/deberta-base-hghl-single-setup-2023-05-13_05-28-06/'
# model_name = './model_config/predict-scm/microsoft/deberta-base-scm-single-setup-2023-05-11_18-55-38/'
# model_name = './model_config/predict-hghl/microsoft/deberta-base-hghl-single-setup-2023-05-22_05-26-15/'

wandb.init(
        project="Predict-"
        +sub_task
        +"-"
        +'microsoft-deberta-base'
        +"-"
        +"-"+task+"-"
        +sub_task+"-split-"
        +str(test_split)
        +"-evalsub- "+eval_subset
        +"-withprtrn"
        +"withmtphrNEW", #nprtrn = no pretraining, withprtrn = with pretrain 
        entity="msen"
    )

model_config = get_model_config(model_name)
max_seq_len = model_config.max_position_embeddings

# print('\n','Using pre-trained checkpoint: ', model_name)
# print('\n','Max sequence length: ', max_seq_len)

def train_model_single_task(itr,
                model_name,
                task='hghl',
                eval_subset='dev',
                output_path='model_config/',
                num_epochs=3, 
                train_batch_size=32, 
                model_suffix='version1', 
                data_file_suffix='', 
                max_seq_length=256, 
                add_special_token=False, 
                loss='MultipleNegativesRankingLoss', 
                add_metaphor = False,
                test_split=0,
                add_lexical_trigger = False, 
                add_xIntent = 0,
                sentence_transformer=True):
    
    ### Configure sentence transformers for training and train on the provided dataset
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    
    path = output_path+specific_output_folder+'/'
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    
    wandb.log({"epochs": num_epochs, "lr": learning_rate, "batch": train_batch_size})

    if not isExist:
        os.makedirs(path)
        print("New directory created: "+specific_output_folder)

    output_path = output_path+specific_output_folder+'/'+model_name+ "-" + model_suffix + "-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if task=='hghl':
        TRAIN_PATH = '../datasets/metaphors-train-part1-clean-csr.csv'
        DEV_PATH = '../datasets/metaphors-dev-part1-clean-csr.csv'  
        TEST_PATH = '../datasets/metaphors-test-part1-clean-csr.csv'

    elif task=='scm':
        TRAIN_PATH = '../datasets/metaphors-train-part1-clean-csr.csv'
        DEV_PATH = '../datasets/metaphors-dev-part1-clean-csr.csv'  
        TEST_PATH = '../datasets/metaphors-test-part1-clean-csr.csv'
        
        '''
        For the scm data splits  
        '''
        if test_split == 1:
            TEST_PATH = '../datasets/metaphor-src-test-1.csv'
            print('Testing on split 1')
        if test_split == 2:
            TEST_PATH = '../datasets/metaphor-src-test-2.csv'
            print('Testing on split 2')
        if test_split == 3:    
            TEST_PATH = '../datasets/metaphor-src-test-3.csv'
            print('Testing on split 3')
        if test_split == 4:    
            TEST_PATH = '../datasets/metaphor-src-test-4.csv'
            print('Testing on split 4')

    else:
        print('No task name inserted')

    if sentence_transformer:
        word_embedding_model = SentenceTransformer(model_name)
        word_embedding_model.max_seq_length = max_seq_length
        
        if add_special_token:
            word_embedding_model.tokenizer.add_tokens(['<SEP>'], special_tokens=True)
            word_embedding_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    else:
        word_embedding_model = models.Transformer(model_name) 
        word_embedding_model.max_seq_length = max_seq_length
    
        if add_special_token:
            word_embedding_model.tokenizer.add_tokens(['<SEP>'], special_tokens=True)
            word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_examples = []
    with open(TRAIN_PATH, encoding="utf-8") as fIn:
        reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if loss=='MultipleNegativesRankingLoss':
                if task=='hghl':
                    if add_metaphor == False:
                        print('AS INTENDED HGHL BLOCK WIHTOUT METAPHOR')
                        train_examples.append(InputExample(texts=[
                            row['Sentence'], 
                            row['Schema Slot']]))
                    else:
                        print('AS INTENDED HGHL BLOCK WIHT METAPHOR')
                        train_examples.append(InputExample(texts=[
                            row['Sentence'] +'<SEP>'+row['Source LM'], 
                            row['Schema Slot']]))
                else:
                    if add_metaphor == False:
                        print('AS INTENDED SCM BLOCK WITHOUT METAPHOR')
                        train_examples.append(InputExample(texts=[
                            row['Sentence'], 
                            row['Source CM']]))
                    else:
                        print('AS INTENDED SCM BLOCK WITH METAPHOR')
                        train_examples.append(InputExample(texts=[
                            row['Sentence']+'<SEP>'+row['Source LM'], 
                            row['Source CM']]))  
            
    if loss == 'MultipleNegativesRankingLoss':
        # Special data loader that avoid duplicates within a batch
        train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=train_batch_size)
        # Our training loss
        train_loss = losses.MultipleNegativesRankingLoss(model)

    if eval_subset=='dev': 
        var = pd.read_csv(DEV_PATH)
    else:
        var = pd.read_csv(TEST_PATH)

    if task=='hghl':
        
        if add_metaphor == False:
            print('INTENDED HGHL DEV BLOCK WITHOUT METAPHOR')
            sentences = var['Sentence']
            pos = var['Schema Slot']
        else:
            print('INTENDED HGHL DEV BLOCK WITH METAPHOR')
            sentences = var['Sentence'] 
            # + '<SEP>' + row['Source LM']
            pos = var['Schema Slot']
    
    else:
        if add_metaphor == False:
            print('INTENDED SCM DEV BLOCK WITHOUT METAPHOR')
            sentences = var['Sentence']
            pos = var['Source CM']
        else:
            print('INTENDED SCM DEV BLOCK WITH METAPHOR')
            sentences = var['Sentence'] 
            # + '<SEP>' + row['Source LM']
            pos = var['Source CM']

    assert len(sentences) == len(pos)

    anchors_with_ground_truth_candidates = dict(zip(list(sentences), list(pos)))
    print('# Length of dev data: ',len(anchors_with_ground_truth_candidates))
    evaluator = RankingEvaluator(itr,
                                model_name,
                                anchors_with_ground_truth_candidates, 
                                task,
                                loss,
                                show_progress_bar=False,
                                name=eval_subset
                                )

    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) # 10% of train data

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              evaluator=evaluator,
              evaluation_steps=100,
              warmup_steps=warmup_steps,
              optimizer_params={'lr': learning_rate},
              save_best_model = True,
              show_progress_bar = True,
              output_path=output_path)

# '''
# Only for sweep stuff
# '''

# sweep_configuration = {
#     'method': 'random',
#     'name': 'sweep',
#     'metric': {'goal': 'maximize', 'name': 'macro_average'},
#     'parameters': 
#     {
#         'batch_size': {'values': [8,16,32]},
#         'epochs': {'values': [4, 5, 6]},
#         'lr': {'max': 0.1, 'min': 0.00001}
#      },
# }

# run = wandb.init(project="check-hyper-sweeps", entity="msen", config=sweep_configuration)
# assert run is wandb.run
# sweep_id = wandb.sweep(sweep=sweep_configuration, project="check-hyper-sweeps")


if task =='single':
    print('Training single-task model')
    '''
    Hyperparameters only for sweeps. Comment otherwise.
    '''
    # print(wandb.config)
    # print(type(wandb.config))

    # num_epochs = wandb.config.parameters['epochs']
    # learning_rate = wandb.config.parameters['lr']
    # train_batch_size = wandb.config.parameters['batch_size']
        
    train_model_single_task(itr,
                model_name, 
                task= sub_task, 
                eval_subset=eval_subset,
                num_epochs=epochs, 
                train_batch_size=train_batch_size, 
                model_suffix=sub_task+'-'+task+'-setup', 
                data_file_suffix='', 
                max_seq_length=512, 
                add_special_token=True, 
                loss='MultipleNegativesRankingLoss', 
                add_metaphor = True,
                test_split = test_split, 
                add_lexical_trigger = False,
                add_xIntent = add_xIntent,
                sentence_transformer=False)
    
    # Start sweep job.
    # wandb.agent(sweep_id, function=train_model_single_task)

# distilbert-base-uncased-single-hghl-split0-batch32-2022-12-30_14-07-41_best_allx_scm_single_dev


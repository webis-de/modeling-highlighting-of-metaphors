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

print('\n', 'Iteration: ', itr, '\n')
print('\n', 'Task: ', task, '\n')
print('\n', 'Epochs: ', epochs, '\n')
print('\n', 'Train Batch Size: ', train_batch_size, '\n')
print('\n', 'Learning Rate: ', learning_rate, '\n')
print('\n', 'Sub Task: ', sub_task, '\n')
print('\n', 'Eval Subset: ', eval_subset, '\n')
print('\n', 'xAll: ', add_xIntent, '\n')
print('\n', 'Specific Output Folder Path: ', specific_output_folder, '\n')

def get_model_config(model_name):
    config = AutoConfig.from_pretrained(model_name)
    return config

# model_name = 'distilbert-base-uncased' 
model_name = 'microsoft/deberta-base' 

wandb.init(
        project="Predict-"
        +sub_task
        +"-"
        +model_name.replace('/','-')
        +"-"
        +"-"+task+"-"
        +sub_task+"-split-"
        +str(test_split)
        +"-evalsub- "
        +eval_subset
        +"-nprtrnwithmtphrNEW", #nprtrn = no pretraining, withprtrn = with pretrain 
        entity="msen"
    )
    
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'batch_size': {'values': [32]},
        'epochs': {'values': [3, 4, 5, 6]},
        'lr': {'max': 0.1, 'min': 0.00001}
    }
}

# sweep_id = wandb.sweep(sweep=sweep_configuration, project='my-second-sweep')

model_config = get_model_config(model_name)
max_seq_len = model_config.max_position_embeddings

# print('\n','Using pre-trained checkpoint: ', model_name)
# print('\n','Max sequence length: ', max_seq_len)

# print('\n','######## Config Overview ########', '\n', model_config)
# print('######## ######## ########')

def train_model_multitask(itr,
                model_name,
                specific_output_folder='',
                task='hghl',
                eval_subset='dev',
                output_path='model_config_multi_task/',
                num_epochs=3, 
                train_batch_size=32, 
                learning_rate=5e-5,
                model_suffix='', 
                data_file_suffix='', 
                max_seq_length=256, 
                add_special_token=False, 
                loss='MultipleNegativesRankingLoss', 
                add_metaphor = False,
                test_split=0,
                add_lexical_trigger = False,
                add_xIntent = False,
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

    TRAIN_PATH_HGHL = '../datasets/metaphors-train-part1-clean-csr.csv'
    DEV_PATH_HGHL = '../datasets/metaphors-dev-part1-clean-csr.csv'  
    TEST_PATH_HGHL = '../datasets/metaphors-test-part1-clean-csr.csv'

    TRAIN_PATH_SCM = '../datasets/metaphors-train-part1-clean-csr.csv'
    DEV_PATH_SCM = '../datasets/metaphors-dev-part1-clean-csr.csv'  
    TEST_PATH_SCM = '../datasets/metaphors-test-part1-clean-csr.csv'
        
    '''
    For the data splits  
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
        print('Not testing on any split')

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
        
        word_embedding_model.tokenizer.add_tokens(['<hghl>', '<scm>'], special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))


    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    print('Pooling model architecture:\n',pooling_model,'\n')
    
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    print('Sentence Transformer architecture:\n',model,'\n')

    train_examples_hghl = []
    train_examples_scm = []
    
    with open(TRAIN_PATH_HGHL, encoding="utf-8") as fIn_hghl:
        reader1 = csv.DictReader(fIn_hghl, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for rowx in reader1:
            if loss=='MultipleNegativesRankingLoss':
                if add_metaphor == True:
                    train_examples_hghl.append(InputExample(texts=[
                        '<hghl>'+rowx['Sentence']+'<SEP>'+rowx['Source LM'], 
                        rowx['Schema Slot']]))
                else:
                    train_examples_hghl.append(InputExample(texts=[
                        '<hghl>'+rowx['Sentence'], 
                        rowx['Schema Slot']]))
    
    with open(TRAIN_PATH_SCM, encoding="utf-8") as fIn_scm:
        reader2 = csv.DictReader(fIn_scm, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for rowy in reader2:
            if loss=='MultipleNegativesRankingLoss':
                if add_metaphor == True:
                    train_examples_scm.append(InputExample(texts=[
                        '<scm>'+rowy['Sentence']+'<SEP>'+rowy['Source LM'], 
                        rowy['Source CM']]))
                else:
                    train_examples_scm.append(InputExample(texts=[
                        '<scm>'+rowy['Sentence'], 
                        rowy['Source CM']])) 

    if loss == 'MultipleNegativesRankingLoss':
        # Special data loader that avoid duplicates within a batch
        train_dataloader_hghl = DataLoader(train_examples_hghl, shuffle=False, batch_size=train_batch_size)
        # Our training loss
        train_loss_hghl = losses.MultipleNegativesRankingLoss(model)

        # Special data loader that avoid duplicates within a batch
        train_dataloader_scm = DataLoader(train_examples_scm, shuffle=False, batch_size=train_batch_size)
        # Our training loss
        train_loss_scm = losses.MultipleNegativesRankingLoss(model)
    
    if eval_subset=='dev': 
        var_hghl = pd.read_csv(DEV_PATH_HGHL)
        var_scm = pd.read_csv(DEV_PATH_SCM)
    else:
        var_hghl = pd.read_csv(TEST_PATH_HGHL)
        var_scm = pd.read_csv(TEST_PATH_SCM)
    
    if task=='hghl':
        if add_metaphor == False:
            # sentences_hghl = '<hghl>'+var_hghl['Sentence']
            sentences_hghl = var_hghl['Sentence'] 
            pos_hghl = var_hghl['Schema Slot']
            assert len(sentences_hghl) == len(pos_hghl)
        else:
            sentences_hghl = var_hghl['Sentence'] 
            # + '<SEP>' + var_hghl['Source LM']
            pos_hghl = var_hghl['Schema Slot']
            assert len(sentences_hghl) == len(pos_hghl)
    
    else:
        if add_metaphor == False:
            sentences_scm = var_scm['Sentence'] 
            pos_scm = var_scm['Source CM']
            assert len(sentences_scm) == len(pos_scm)
        else:
            sentences_scm = var_scm['Sentence'] 
            # + '<SEP>' + var_hghl['Source LM']
            pos_scm = var_scm['Source CM']
            assert len(sentences_scm) == len(pos_scm)
            
    if task=='hghl':
        anchors_with_ground_truth_candidates_hghl = dict(zip(list(sentences_hghl), list(pos_hghl)))
        evaluator = RankingEvaluator(itr,
                                    model_name,
                                    anchors_with_ground_truth_candidates_hghl, 
                                    task,
                                    loss,
                                    show_progress_bar=True,
                                    name=eval_subset
                                    )
    else:
        anchors_with_ground_truth_candidates_scm = dict(zip(list(sentences_scm), list(pos_scm)))
        print('# Test data: ',len(anchors_with_ground_truth_candidates_scm))
        evaluator = RankingEvaluator(itr,
                                    model_name,
                                    anchors_with_ground_truth_candidates_scm, 
                                    task,
                                    loss,
                                    show_progress_bar=True,
                                    name=eval_subset
                                    )

    warmup_steps = int(len(train_dataloader_scm) * num_epochs * 0.1) # 10% of train data look into this later

    '''
    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0
            ):
    '''

    # Train the model
    model.fit(train_objectives=[(train_dataloader_hghl, train_loss_hghl),(train_dataloader_scm, train_loss_scm)],
              epochs=num_epochs,
              evaluator=evaluator,
              evaluation_steps=100,
              warmup_steps=warmup_steps,
              optimizer_params={'lr': learning_rate},
              save_best_model = True,
              show_progress_bar=True,
              output_path=output_path)

print('Training multi-task model')

train_model_multitask(itr,
            model_name, 
            specific_output_folder=specific_output_folder,
            task= sub_task, 
            eval_subset=eval_subset,
            num_epochs=epochs, 
            train_batch_size=train_batch_size, 
            learning_rate = learning_rate,
            model_suffix='multi-all-'+sub_task, 
            data_file_suffix='', 
            max_seq_length=512, 
            add_special_token=True, 
            loss='MultipleNegativesRankingLoss', 
            add_metaphor = True,
            test_split = test_split, 
            add_lexical_trigger = False,
            add_xIntent = add_xIntent,
            sentence_transformer=False)

# wandb.agent(sweep_id, function=train_model_multitask, count=4)
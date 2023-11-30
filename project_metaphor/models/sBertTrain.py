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
args = parser.parse_args()

itr = args.itr
task = args.task

print('\n', 'Iteration: ', itr, '\n')
print('\n', 'Task: ', task, '\n')

wandb.init(project="mpnet-single-scm-split-1-allx", entity="msen")

def get_model_config(model_name):
    config = AutoConfig.from_pretrained(model_name)
    return config

# model_name = 'bert-base-uncased' # works
# model_name = 'distilbert-base-uncased' # works
# model_name = 'bert-large-uncased' # works with batch size 16 all acc 0.00 - 0.04
model_name = 'sentence-transformers/all-mpnet-base-v2' # works
# model_name = 'albert-base-v2' # works
# model_name = 'xlm-roberta-base' # all acc 0.00
# model_name = 'roberta-large' # works 
# model_name = 'albert-base-v2'

model_config = get_model_config(model_name)
max_seq_len = model_config.max_position_embeddings

print('\n','Using pre-trained checkpoint: ', model_name)
print('\n','Max sequence length: ', max_seq_len)

print('\n','######## Config Overview ########', '\n', model_config)
print('######## ######## ########')

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
                add_xIntent = False,
                sentence_transformer=True):

    ### Configure sentence transformers for training and train on the provided dataset
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    output_path = output_path+model_name+ "-" + model_suffix + "-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if task=='hghl':
        TRAIN_PATH = '../datasets/metaphor-train.csv'
        DEV_PATH = '../datasets/metaphor-dev.csv'
        TEST_PATH = '../datasets/metaphor-test.csv'

    elif task=='scm':
        TRAIN_PATH = '../datasets/metaphors-train-part1-clean-csr.csv'
        DEV_PATH = '../datasets/metaphor-scm-dev-part1.csv'
        TEST_PATH = '../datasets/metaphor-scm-test-part1.csv'
        
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
    # Clean CSR_xWant,
    # Clean CSR_xIntent,
    # Clean CSR_xAttr,
    # Clean CSR_xReact,
    # Clean CSR_xEffect,
    # Clean CSR_xNeed,
    # Clean CSR_xReason
    with open(TRAIN_PATH, encoding="utf-8") as fIn:
    
        reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            
            if loss=='MultipleNegativesRankingLoss':
                
                if task=='hghl':
                    if add_lexical_trigger == False:
                        train_examples.append(InputExample(texts=[
                            row['Sentence'], 
                            row['Schema Slot']])) if add_metaphor == False else \
                        train_examples.append(InputExample(texts=[
                            row['Sentence']+'<SEP>'+
                            row['Source LM'], 
                            row['Schema Slot']]))
                    else:
                        train_examples.append(InputExample(texts=[
                            row['Sentence']+'<SEP>'+
                            row['Lexical Trigger'], 
                            row['Schema Slot']])) if add_metaphor == False else \
                        train_examples.append(InputExample(texts=[
                            row['Sentence']+'<SEP>'+
                            row['Source LM']+'<SEP>'+
                            row['Lexical Trigger'], 
                            row['Schema Slot']]))
                
                else:
                    if add_lexical_trigger == False:
                        # print('\nMetaphor = True\nLexical Trigger = False\nxIntent = True')
                        train_examples.append(InputExample(texts=[
                            row['Sentence']+'<SEP>'+
                            row['Source LM'], 
                            row['Source CM']])) if add_xIntent == False else \
                        train_examples.append(InputExample(texts=[
                            row['Sentence']+'<SEP>'+
                            row['Source LM']+'<SEP>'+
                            row['Clean CSR_xWant']+'<SEP>'+
                            row['Clean CSR_xIntent']+'<SEP>'+
                            row['Clean CSR_xAttr']+'<SEP>'+
                            row['Clean CSR_xReact']+'<SEP>'+
                            row['Clean CSR_xEffect']+'<SEP>'+
                            row['Clean CSR_xNeed']+'<SEP>'+
                            row['Clean CSR_xReason'], 
                            row['Source CM']]))
                
                    else:
                        train_examples.append(InputExample(texts=[
                            row['Sentence']+'<SEP>'+ 
                            row['Lexical Trigger'], 
                            row['Source CM']])) if add_metaphor == False else \
                        train_examples.append(InputExample(texts=[
                            row['Sentence']+'<SEP>'+
                            row['Source LM']+'<SEP>'+
                            row['Lexical Trigger'], 
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
        if add_lexical_trigger == False:
            sentences = var['Sentence'] if add_metaphor == False else \
                        var['Sentence'] + '<SEP>' + \
                        var['Source LM']
            pos = var['Schema Slot']
        else: 
            sentences = var['Sentence'] + '<SEP>' + \
                        var['Lexical Trigger'] if add_metaphor == False else \
                        var['Sentence'] + '<SEP>' + \
                        var['Source LM'] + '<SEP>' + \
                        var['Lexical Trigger']
            pos = var['Schema Slot']
    
    else:
        if add_lexical_trigger == False:
            sentences = var['Sentence'] if add_metaphor == False else \
            var['Sentence'] + '<SEP>' + \
            var['Source LM']
            pos = var['Source CM']
        else: 
            sentences = var['Sentence'] + '<SEP>' + var['Lexical Trigger'] if add_metaphor == False else \
            var['Sentence'] + '<SEP>' + \
            var['Source LM'] + '<SEP>' + \
            var['Lexical Trigger']
            pos = var['Source CM']

    assert len(sentences) == len(pos)

    anchors_with_ground_truth_candidates = dict(zip(list(sentences), list(pos)))
    print('# Test data: ',len(anchors_with_ground_truth_candidates))
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
              optimizer_params={'lr': 5e-5},
              save_best_model = True,
              show_progress_bar = True,
              output_path=output_path)

def train_model_multitask(itr,
                model_name,
                task='hghl',
                eval_subset='dev',
                output_path='model_config/',
                num_epochs=3, 
                train_batch_size=32, 
                model_suffix='', 
                data_file_suffix='', 
                max_seq_length=256, 
                add_special_token=False, 
                loss='MultipleNegativesRankingLoss', 
                add_metaphor = False,
                test_split=0,
                add_lexical_trigger = False,
                sentence_transformer=True):

    ### Configure sentence transformers for training and train on the provided dataset
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    output_path = output_path+model_name+ "-" + model_suffix + "-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    
    TRAIN_PATH_HGHL = '../datasets/metaphor-train.csv'
    DEV_PATH_HGHL = '../datasets/metaphor-dev.csv'
    TEST_PATH_HGHL = '../datasets/metaphor-test.csv'

    TRAIN_PATH_SCM = '../datasets/metaphors-train-part1-clean-csr.csv'
    DEV_PATH_SCM = '../datasets/metaphor-scm-dev-part1.csv'
    TEST_PATH_SCM = '../datasets/metaphor-scm-test-part1.csv'
        
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
    print('Pooling model architecture:\n',pooling_model,'\n')
    
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    print('Sentence Transformer architecture:\n',model,'\n')

    train_examples_hghl = []
    train_examples_scm = []
    
    with open(TRAIN_PATH_HGHL, encoding="utf-8") as fIn_hghl:
        reader1 = csv.DictReader(fIn_hghl, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for rowx in reader1:
            if loss=='MultipleNegativesRankingLoss':
                train_examples_hghl.append(InputExample(texts=[rowx['Sentence'], rowx['Schema Slot']])) if add_metaphor == False else \
                train_examples_hghl.append(InputExample(texts=[rowx['Sentence']+'<SEP>'+rowx['Source LM'], rowx['Schema Slot']]))
    
    with open(TRAIN_PATH_SCM, encoding="utf-8") as fIn_scm:
        reader2 = csv.DictReader(fIn_scm, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for rowy in reader2:
            if loss=='MultipleNegativesRankingLoss':
                train_examples_scm.append(InputExample(texts=[rowy['Sentence'], rowy['Source CM']])) if add_metaphor == False else \
                train_examples_scm.append(InputExample(texts=[
                rowy['Sentence']+'<SEP>'+
                rowy['Source LM']+'<SEP>'+
                rowy['Clean CSR_xWant']+'<SEP>'+
                rowy['Clean CSR_xIntent']+'<SEP>'+
                rowy['Clean CSR_xAttr']+'<SEP>'+
                rowy['Clean CSR_xReact']+'<SEP>'+
                rowy['Clean CSR_xEffect']+'<SEP>'+
                rowy['Clean CSR_xNeed']+'<SEP>'+
                rowy['Clean CSR_xReason'], 
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
        sentences_hghl = var_hghl['Sentence'] if add_metaphor == False else \
                        var_hghl['Sentence'] + '<SEP>' + var_hghl['Target LM']
        pos_hghl = var_hghl['Schema Slot'] 
    else:
        sentences_scm = var_scm['Sentence'] if add_metaphor == False else \
                        var_scm['Sentence'] + '<SEP>' + var_scm['Source LM']
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
              optimizer_params={'lr': 5e-5},
              save_best_model = True,
              show_progress_bar=True,
              output_path=output_path)

if task =='single':
    print('Training single-task model')    
    train_model_single_task(itr,
                model_name, 
                task= 'scm', 
                eval_subset='test',
                num_epochs=5, 
                train_batch_size=32, 
                model_suffix='batch32allx', 
                data_file_suffix='', 
                max_seq_length=512, 
                add_special_token=False, 
                loss='MultipleNegativesRankingLoss', 
                add_metaphor = True,
                test_split =1, 
                add_lexical_trigger = False,
                add_xIntent = True,
                sentence_transformer=False)

else:
    print('Training multi-task model')
    train_model_multitask(itr,
                model_name, 
                task= 'scm', 
                eval_subset='test',
                num_epochs=5, 
                train_batch_size=32, 
                model_suffix='', 
                data_file_suffix='', 
                max_seq_length=512, 
                add_special_token=False, 
                loss='MultipleNegativesRankingLoss', 
                add_metaphor = True,
                test_split =0, 
                add_lexical_trigger = False,
                sentence_transformer=False)


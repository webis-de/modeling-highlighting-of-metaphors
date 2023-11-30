from inference_evaluator_all_accuracies import InferenceRankingEvaluator
from sentence_transformers import SentenceTransformer
import pandas as pd
import logging

setup = int(input("Setup [Options: 0 for 'single' or 1 for 'multi']:\n")) 
test_split = int(input("Test Split [Options: 0 - 4]:\n"))
train_batch_size = int(input("Enter Train Batch Size [Options: 16 - 32]:\n"))
add_xIntent = 0
task = int(input("Enter downstream task: [Options: 0 for highlighted aspects and 1 for Source Domains]:\n"))
hghl_feat = int(input("Use highlighted aspects as features to predict source domains? [Options: 0 for no and 1 for yes]:\n"))

# task = 'hghl' if task == 0 else 'scm'
task = 'hghl'

logging.info('Setup: ' + str(setup))
logging.info('Test Split: ' + str(test_split))
logging.info('Train Batch Size: ' + str(train_batch_size))
logging.info('CSR: ' + str(add_xIntent))
logging.info('Downstream Task: ' + str(task))
logging.info('Use highlighted aspects as features: ' + str(hghl_feat))

TEST_PATH = '../datasets/metaphors-test-part1-clean-csr.csv'

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

var = pd.read_csv(TEST_PATH)

best_without_scm = './model_config/predict-hghl/distilbert-base-uncased-hghl-single-task-2023-04-25_14-15-20/'
best_without_scm_multi = './model_config_multi_task/predict-hghl/distilbert-base-uncased-multi-all-hghl-2023-05-10_06-17-30/'
best_with_scm = './/'

'''
###### Best of source domain prediction validation set ######
'''

without_hghl_multi = './model_config_multi_task/predict-scm/distilbert-base-uncased-multi-all-scm-2023-04-26_03-32-09/'
without_hghl = './model_config/predict-scm/distilbert-base-uncased-scm-single-task-2023-04-22_14-30-35/'
with_hghl = './model_config/predict-scm/./model_config/SingleHghlAll/distilbert-base-uncased-single-hghl-all-2023-03-08_19-10-56/-hghl-single-task-2023-05-09_01-10-52/'

if setup == 0: # 0 for single 
    # logging.info('In single task setup')
    if task == 'scm':
        logging.info('In single SCM setup')
        model = SentenceTransformer(with_hghl)
        sentences = var['Sentence'] + '<SEP>' + var['Source LM']
        pos = var['Source CM']
    else:
        logging.info('In single HGHL setup')
        # model = SentenceTransformer(best_single_without_csr_hghl)
        model = SentenceTransformer(best_without_scm)
        # model = SentenceTransformer(best_with_scm)
        sentences = var['Sentence'] + '<SEP>' + var['Source LM']
        pos = var['Schema Slot']

    assert len(sentences) == len(pos)

else:
    # logging.info('In multi task setup')
    if task == 'scm':
        logging.info('In no csr block of source cm multi')
        model = SentenceTransformer(without_hghl_multi)
        # sentences = '<scm>'+'<SEP>'+var['Sentence']
        sentences = var['Sentence'] + '<SEP>' + var['Source LM']
        pos = var['Source CM']
    else:
        print('where multi hghl i want')
        model = SentenceTransformer(best_without_scm_multi)
        sentences = var['Sentence'] + '<SEP>' + var['Source LM']
        pos = var['Schema Slot']

    assert len(sentences) == len(pos)

anchors_with_ground_truth_candidates = dict(zip(list(sentences), list(pos)))

result = InferenceRankingEvaluator(
                                model,
                                anchors_with_ground_truth_candidates, 
                                task)
result()

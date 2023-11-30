from inference_evaluator_all_accuracies import InferenceRankingEvaluator
from sentence_transformers import SentenceTransformer
import pandas as pd
import logging

setup = int(input("Setup [Options: 0 for 'single' or 1 for 'multi']:\n")) 
test_split = int(input("Test Split [Options: 0 - 4]:\n"))
train_batch_size = int(input("Enter Train Batch Size [Options: 16 - 32]:\n"))
add_xIntent = 0
task = int(input("Enter downstream task: [Options: 0 for highlighted aspects and 1 for Source Domains]:\n"))
hghl_feat = task = int(input("Use highlighted aspects as features to predict source domains? [Options: 0 for no and 1 for yes]:\n"))

# task = 'hghl' if task == 0 else 'scm'
task = 'scm'

logging.info('Setup: ' + str(setup))
logging.info('Test Split: ' + str(test_split))
logging.info('Train Batch Size: ' + str(train_batch_size))
logging.info('CSR: ' + str(add_xIntent))
logging.info('Downstream Task: ' + str(task))
logging.info('Use highlighted aspects as features: ' + str(hghl_feat))

TEST_PATH = '../datasets/metaphors-test-part1-clean-csr.csv'

if test_split == 1:
    TEST_PATH = '../datasets/hghl-test-less-than-ten.csv'
    print('Testing on split 1')
if test_split == 2:
    TEST_PATH = '../datasets/hghl-test-more-than-ten.csv'
    print('Testing on split 2')

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
'''


var = pd.read_csv(TEST_PATH)

'''
###### Best of highlighted aspects prediction validation set ######
'''

# best_without_scm = './model_config/predict-hghl/distilbert-base-uncased-hghl-single-task-2023-04-15_14-22-11/'
# best_without_scm_multi = './model_config_multi_task/MultiAll/distilbert-base-uncased-multi-all-hghl-2023-02-25_21-57-17/'
# best_with_scm = './model_config/predict-hghl/./model_config/predict-scm/distilbert-base-uncased-hghl-single-task-2023-04-17_04-52-51/-hghl-single-task-2023-05-04_04-19-04/'

'''
###### Best of source domain prediction validation set ######
'''

'''
distilbert
'''

# without_hghl_multi = './model_config_multi_task/predict-scm/distilbert-base-uncased-multi-all-scm-2023-04-20_16-08-43/'
# without_hghl = './model_config/predict-scm/distilbert-base-uncased-hghl-single-task-2023-04-17_04-52-51/'
# with_hghl = './model_config/predict-scm/./model_config/predict-hghl/distilbert-base-uncased-hghl-single-task-2023-04-15_14-22-11/-scm-single-task-2023-04-20_05-40-09/'

# best_single_without_csr_hghl = './model_config/distilbert-base-uncased-single-hghl-split0-batch32-2022-12-31_12-27-18/'
# best_single_with_csr_hghl = 'model_config/distilbert-base-uncased-single-hghl-split0-batch32-2022-12-31_13-57-01'

# with or without highlighted aspects used as features to predict source domains
# if hghl_feat == 1:
#     without_csr_scm = './model_config/distilbert-base-uncased-single-hghl-split0-batch32-2023-01-04_14-41-28/'
#     with_csr_scm = 'model_config/distilbert-base-uncased-single-hghl-split0-batch32-2022-12-31_13-57-01' 
 
# best_multi_without_csr_hghl = 'model_config/distilbert-base-uncased-multi-hghl-split0-batch32-2022-12-31_13-09-26'
# best_multi_with_csr_hghl = 'model_config/distilbert-base-uncased-multi-hghl-split0-batch32-2022-12-31_14-24-36'

# checkpoint_path = './model_config/hghl_new_setup_optm_epoch/'


'''
deberta
'''

'''
Predict SCM
'''
# without_hghl = './model_config/predict-scm/microsoft/deberta-base-scm-single-setup-2023-05-11_18-55-38/' 
without_hghl_multi = './model_config_multi_task/predict-scm/microsoft/deberta-base-multi-all-scm-2023-06-05_18-35-41/'
# with_hghl = './model_config/predict-scm/./model_config/predict-hghl/microsoft/deberta-base-hghl-single-setup-2023-05-13_05-28-06/-scm-single-setup-2023-05-16_01-32-38/'

'''
Predict Hghl
'''
best_without_scm =  './model_config/predict-hghl/microsoft/deberta-base-hghl-single-setup-2023-05-13_05-28-06/'
best_without_scm_multi = './model_config_multi_task/predict-hghl/microsoft/deberta-base-multi-all-hghl-2023-05-12_20-34-36/' 
best_with_scm = './model_config/predict-hghl/./model_config/predict-scm/microsoft/deberta-base-scm-single-setup-2023-05-11_18-55-38/-hghl-single-setup-2023-05-18_07-56-13/'

if setup == 0: # 0 for single 
    # logging.info('In single task setup')
    if task == 'scm':
        logging.info('In single SCM setup')
        model = SentenceTransformer(with_hghl)
        sentences = var['Sentence']
        pos = var['Source CM']
    else:
        logging.info('In single HGHL setup')
        model = SentenceTransformer(best_with_scm)
        # model = SentenceTransformer(best_without_scm)
        sentences = var['Sentence']
        pos = var['Schema Slot']

    assert len(sentences) == len(pos)

else:
    # logging.info('In multi task setup')
    if task == 'scm':
        logging.info('In no csr block of source cm multi')
        model = SentenceTransformer(without_hghl_multi)
        # sentences = '<scm>'+'<SEP>'+var['Sentence']
        sentences = var['Sentence']
        pos = var['Source CM']
    else:
        print('where multi hghl i want')
        model = SentenceTransformer(best_without_scm_multi)
        sentences = var['Sentence']
        pos = var['Schema Slot']

    assert len(sentences) == len(pos)

anchors_with_ground_truth_candidates = dict(zip(list(sentences), list(pos)))

result = InferenceRankingEvaluator(
                                model,
                                anchors_with_ground_truth_candidates, 
                                task)
result()


'''

298,298,298,1581,"Representative democracy was in fact born of many and 
different power conflicts, many of them bitterly fought in opposition to 
ruling groups, whether they were church hierarchies, landowners or imperial 
monarchies, often in the name of 'the people'.",
10 Human Life Cycle and Family Relations,Entity,Representative democracy,
[entity] was * born,born,Representative democracy,Democracy,born,Life Stage,2377351.0,,,
[entity] was * born,entity was * born,"[[' born', ' curious', ' intelligent']]",
"[[' PersonX is born.', ' PersonX is born', ' none']]","[[' none', ' to be born', ' to be a person']]",
"[[' none', ' to be born', ' to have been born']]","[[' happy', ' happy.', ' excited']]","[[' PersonX is born', 
' PersonX was born', ' PersonX is born into the world']]","[[' to have children', ' to have a family', ' to be a good person']]",
"Want: to have children,  to be a good person,  to have a family","Intents:  to be born,  to be a person","Attributes:  curious,  intelligent, born","Reactions:  happy, happy,  excited","Effects: speaker is born,  speaker is born","Needs:  to be born,  to have been born","Reasons: speaker is born,  speaker was born,  speaker is born into the world"

'''

# conflicts

'''

1. Analysis idea 1 --> We see stuff that never made it to even best of 5 
2. Analysis idea 2 --> Confusion matrix 

'''
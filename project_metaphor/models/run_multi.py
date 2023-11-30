'''
@All OS commands to run models for 20 iteraitons 
'''

import os

''' 
    run sBERT with Bert 
'''

epochs = input("Epochs [Options: 0 - 6]:\n")
train_batch_size = input("Enter Train Batch Size [Options: 16 - 32]:\n")
add_xIntent = input("xAll [0 or 1]:\n")
test_split = input("Enter test data [Options: 0 - 4]:\n")
sub_task = input("Enter downstream task: [Options: 'hghl' for highlighted aspects and 'scm' for Source Domains]:\n")
eval_subset = input("Enter eval subset [Options: 'dev' or 'test']:\n") 

for i in range(20):
    os.system("CUDA_LAUNCH_BLOCKING=1 python3 \
    sBertTrain2.py \
    {0} multi {1} {2} {3} {4} {5} {6}"\
    .format(i, \
        epochs, \
        train_batch_size, \
        add_xIntent, \
        test_split, \
        sub_task, \
        eval_subset))
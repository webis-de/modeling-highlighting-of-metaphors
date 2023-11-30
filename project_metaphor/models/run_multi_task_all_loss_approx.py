'''
@All OS commands to run models for 20 iteraitons 
'''

import os

''' 
    run sBERT Multi Task Experiment with Bert-based encoders for all parameters 
'''

# epochs = input("Epochs [Options: 0 - 6]:\n")
# train_batch_size = input("Enter Train Batch Size [Options: 16 - 32]:\n")
# add_xIntent = input("xAll [0 or 1]:\n")
# test_split = input("Enter test data [Options: 0 - 4]:\n")
# sub_task = input("Enter downstream task: [Options: 'hghl' for highlighted aspects and 'scm' for Source Domains]:\n")
# eval_subset = input("Enter eval subset [Options: 'dev' or 'test']:\n")
# specific_output_folder = input("Enter specifi output folder path [Example: 'banana' or 'distillbert6epochexperiment']:\n")

# for i in range(20):
#     os.system("CUDA_LAUNCH_BLOCKING=1 python3 \
#     sBertTrainMultiTask.py \
#     {0} single {1} {2} {3} {4} {5} {6} {7}"\
#     .format(i, \
#         epochs, \
#         train_batch_size, \
#         add_xIntent, \
#         test_split, \
#         sub_task, \
#         eval_subset, \
#         specific_output_folder))

epochs = [4,5,6]
learning_rate = [2e-5, 3e-5, 4e-5,5e-5]
batch_size = [8, 16, 32]

for epoch in epochs:
    for lr in learning_rate:
        for batch in batch_size:
            for i in range(20):
                os.system("CUDA_LAUNCH_BLOCKING=1 python3 \
                    sBertMultiLossApprox.py \
                    {0} \
                    multi \
                    {1} \
                    {2} \
                    {3} \
                    0 \
                    0 \
                    scm \
                    dev \
                    predict-scm"\
                    .format(i, \
                        epoch, \
                        batch, \
                        lr
                        ))
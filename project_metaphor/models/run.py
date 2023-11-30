'''
@All OS commands to run models for 20 iteraitons 
'''

import os

''' 
    run sBERT with Bert 
'''

for i in range(20):
    os.system("CUDA_LAUNCH_BLOCKING=1 python3 sBertTrain.py {} single".format(i))


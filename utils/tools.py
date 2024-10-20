import os
import random
import re

import numpy as np
import scipy.stats as stats
import torch
from scipy.stats import ttest_1samp


def setSeed(seed):
   os.environ['PYTHONHASHSEED'] = str(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True
   np.random.seed(seed)
   random.seed(seed)


def getScore(response):
    # match = re.search(r'(\d+)', response)
    match = re.search(r'(\d+(\.\d+)?)', response)
    if match:
        score = match.group(1)
        return score
    else:
        return -1
    

def getSummary(soces:list=None):
    ss = []
    failNum = 0
    for s in soces:
        if s == -1:
            failNum += 1
        else:
            ss.append(s)
    
    avg = np.mean(ss).round(2)

    return f"Test Num: {len(soces)}, UnMatch Num: {failNum}, Average Score(except UnMatch): {avg}."


def getTandPValues(input1:np.ndarray, input2:np.ndarray):
    
    sample = input1 - input2
    t_statistic, p_value = ttest_1samp(sample, 0)
    # t_statistic, p_value = stats.ttest_ind(input1, input2)

    return t_statistic.round(3), p_value.round(3)
import os
import random
import re

import numpy as np
import torch


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
    match = re.search(r'(\d+)', response)
    if match:
        score = match.group(1)
        return score
    else:
        return -1
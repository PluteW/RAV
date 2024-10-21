import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from Evaluate.EvalHCT import getHCTSummary
from Evaluate.EvalSSVT import getSSVTDummary
from utils.Logger import logger, printLog
from utils.tools import getSummary


def getEvaluationForVTL(model:str=""):
    htc_list = getHCTSummary(model=model)
    ssvt_list = getSSVTDummary(model=model)

    tvl_list = htc_list + ssvt_list

    result = getSummary(tvl_list)

    printLog(f"Reult on VTL: \n\t{result}", logger)


if __name__ == "__main__":
    
    model = "SyncVote"

    # model = "GPT4V"

    # model = "TVL-TINY"
    # model = "TVL-S"
    # model = "TVL-B"
    # model = "TVL-BGS"
    getEvaluationForVTL(model)
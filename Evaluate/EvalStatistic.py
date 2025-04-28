import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import json

import numpy as np

from utils.Logger import logger, printLog
from utils.tools import getTandPValues


def judgeScoresInStatics(inputfile1: str, inputfile2: str):
    
    with open(inputfile1, 'r', encoding='utf-8') as json_file:
        loaded_scores_1 = json.load(json_file)
    
    with open(inputfile2, 'r', encoding='utf-8') as json_file:
        loaded_scores_2 = json.load(json_file)
    
    scores_1 = []
    scores_2 = []

    for key, value in loaded_scores_1.items():
        scores_1.append(value["score"])
    
    for key, value in loaded_scores_2.items():
        scores_2.append(value["score"])

    scores_1 = np.array(scores_1)
    scores_2 = np.array(scores_2)
    
    t_statistic, p_value = getTandPValues(scores_1, scores_2)

    printLog(f"t_statistic :{t_statistic}, p_value: {p_value}", logger)


if __name__ == "__main__":

    # file1 = "/home/aa/Desktop/WJL/VTRAG/Evaluate/EvalResult/SSVTP/GPT4V-gpt-4-BatchFormatOutput.json"
    
    # file2 = "/home/aa/Desktop/WJL/VTRAG/Evaluate/EvalResult/SSVTP/SyncVote-gpt-4-BatchFormatOutput.json"

    # file2 = "/home/aa/Desktop/WJL/VTRAG/Evaluate/EvalResult/SSVTP/TVL-B-gpt-4-BatchFormatOutput.json"

    # file2 = "/home/aa/Desktop/WJL/VTRAG/Evaluate/EvalResult/SSVTP/TVL-BGS-gpt-4-BatchFormatOutput.json"

    # file2 = "/home/aa/Desktop/WJL/VTRAG/Evaluate/EvalResult/SSVTP/TVL-S-gpt-4-BatchFormatOutput.json"

    file1 = "/home/aa/Desktop/WJL/VTRAG/Evaluate/EvalResult/HCT/GPT4V-gpt-4-BatchFormatOutput.json"

    file2 = "/home/aa/Desktop/WJL/VTRAG/Evaluate/EvalResult/HCT/SyncVote-gpt-4-BatchFormatOutput-Checked.json"

    judgeScoresInStatics(file1, file2)
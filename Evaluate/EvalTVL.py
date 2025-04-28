import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import os
import sys

from Evaluate.EvalHCT import getHCTSummary
from Evaluate.EvalSSVTP import getSSVTPummary
from utils.Logger import logger, printLog
from utils.tools import getSummary

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import json
import logging
import time

from torch.utils.data import DataLoader, Dataset

from Config.Config import getConfigFromYaml
from Dataset.HCTDS import HCTDS
from Dataset.SSVTPDS import SSVTPDS
from Evaluate.BatchEval import conductBatchInputFile, formatBatchOutputFile
from Model.DualVote.DualVote import DualVote
from Model.Model import Model
from Model.SyncVote import SyncVote
from Model.WeightVote import WeightVote
from utils.EvalFunctions import (
    EVAL_PROMPT,
    SYSTEM_PROMPT,
    get_LLama_evaluator,
    get_Proxy_evaluator,
    get_QWen_evaluator,
)
from utils.Logger import logger, printLog
from utils.Path import EVAL_RESULT_PATH, MISSION_DESCRIPTION
from utils.tools import getJson, getScore, getSummary, setSeed

basicConfigPath = "/home/aa/Desktop/WJL/VTRAG/Config/Config.yaml"

def getEvaluationForTVL(model:str=""):
    htc_list = getHCTSummary(model=model)
    ssvt_list = getSSVTPummary(model=model)

    tvl_list = htc_list + ssvt_list

    result = getSummary(tvl_list)

    printLog(f"Reult on TVL: \n\t{result}", logger)


def TestModelOn(reponses:dict=None, datasets:Dataset=None, dataloader:DataLoader=None, model:Model=None, startId:int=0):

    assert model != None, "You should input a accessible model object!"

    assert datasets != None, "You missed a dataset!"

    assert dataloader != None, "You missed a dataloader!"

    for i, item in zip(range(len(datasets)),dataloader):
        visionPath = item[0][0]
        touchPath = item[1][0]
        
        assistant_response = model.answer(visionPath, touchPath)
        groundTruth = item[2][0].strip().replace(" ","")

        printLog(f"Item index {i}\n\tGROUND TRUTH: {groundTruth}, ASSISTANT: {assistant_response}", logger)
        
        item = {
            "groundTruth": groundTruth,
            "response": assistant_response,
        }
        
        id = int(i) + startId

        reponses[id] = item

        printLog("*"*100+"\n", logger)

    return reponses

def TestOnDatasets(model: Model=None, mission: str=MISSION_DESCRIPTION, datasetsName: str="", gptModel: str=""):

    startId = 0
    responses = {}
    if datasetsName == "TVL" or datasetsName == "SSVTP":
        setSeed(21)
        datasets = SSVTPDS("test", None, mission)
        dataloader = DataLoader(datasets,shuffle=True, batch_size=1)
        
        responses = TestModelOn(responses, datasets, dataloader, model, startId)
        
        startId = len(datasets)
    
    if datasetsName == "TVL" or datasetsName == "HCT":
        setSeed(21)
        datasets = HCTDS("test", None, mission)
        dataloader = DataLoader(datasets,shuffle=True, batch_size=1)

        responses = TestModelOn(responses, datasets, dataloader, model, startId)

    filePath = f"{EVAL_RESULT_PATH}/{datasetsName}/{model.name}.json"
    with open(filePath, 'w', encoding='utf-8') as json_file:
            json.dump(responses, json_file, ensure_ascii=False, indent=4)
    
    conductBatchInputFile(datasetsName, model.name, gptModel)


if __name__ == "__main__":

    # model = "SyncVote"

    # model = "GPT4V"

    # model = "TVL-TINY"
    # model = "TVL-S"
    # model = "TVL-B"
    # model = "TVL-BGS"

    # getEvaluationForTVL(model)

    configPath = "/home/aa/Desktop/WJL/VTRAG/Config/Config.yaml"

    config = getConfigFromYaml(configPath)
    if config.Model.name == "SyncVote":
      model = SyncVote(config.Model.args, config.mission)
    elif config.Model.name == "DualVote":
        model = DualVote(config.Model.args, config.mission)
    elif config.Model.name == "WeightVote":
        model = WeightVote(config.Model.args, config.mission)

    TestOnDatasets(model=model, mission=config.mission, datasetsName=config.dataset, gptModel=config.EVAL_PROXY_MODEL)
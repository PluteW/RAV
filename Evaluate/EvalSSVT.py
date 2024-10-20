import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import json
import time

from PIL import Image
from torch.utils.data import DataLoader

from Config.Config import getConfigFromYaml
from Dataset.SSVTPDS import SSVTPDS
from Evaluate.BatchEvalSSVT import conductBatchInputFile, formatBatchOutputFile
from Model.Model import Model
from Model.SyncVote import SyncVote
from utils.EvalFunctions import (
    EVAL_PROMPT,
    SYSTEM_PROMPT,
    get_LLama_evaluator,
    get_Proxy_evaluator,
    get_QWen_evaluator,
)
from utils.Logger import logger, printLog
from utils.Path import EVAL_RESULT_PATH, MISSION_DESCRIPTION
from utils.tools import getScore, getSummary, setSeed

basicConfigPath = "/home/aa/Desktop/WJL/VTRAG/Config/Config.yaml"

def TestOnSSVTDataset(
        model: Model=None,
        dataloader:DataLoader=None,
        batchEval:bool=True,
        eval_fn:callable=None,
        configPath:str=basicConfigPath, 
        mission:str=MISSION_DESCRIPTION, 
        save=False,
        reTest=False,
    ):

    filePath = f"{EVAL_RESULT_PATH}/SSVT/{model.name}.json"

    if reTest == False:
        if os.path.exists(filePath):
            with open(filePath, 'r', encoding='utf-8') as json_file:
                loaded_scores = json.load(json_file)
            score_list = [entry["score"] for entry in loaded_scores.values()]

            printLog(f"Load result success from: {filePath}. Skip the details of test", logger)

            result = getSummary(score_list)
            
            printLog(f"Reult: \n\t{result}", logger)

            return

        else:
            Warning(f"We could not find the result json file with path: {filePath}! We will continue to process the test again.")
            printLog(f"We could not find the result json file with path: {filePath}! We will continue to process the test again.", logger)
            time.sleep(3)
        
    assert model != None, "You should input a accessible model object!"

    if dataloader == None:
        setSeed(21)
        datasets = SSVTPDS("test", None, mission)
        dataloader = DataLoader(datasets,shuffle=True, batch_size=1)

    config = getConfigFromYaml(configPath)

    if not batchEval:
        if eval_fn == None:
            # eval_fn = get_QWen_evaluator(config, EVAL_PROMPT)
            # eval_fn = get_LLama_evaluator(config, EVAL_PROMPT)
            eval_fn = get_Proxy_evaluator(config, EVAL_PROMPT, SYSTEM_PROMPT)

            score_list = []

    scores = {}
    
    for i, item in zip(range(len(datasets)),dataloader):
        visionPath = item[0][0]
        touchPath = item[1][0]
        
        assistant_response = model.answer(visionPath, touchPath)
        groundTruth = item[2][0].strip().replace(" ","")

        printLog(f"Item index {i}\n\tGROUND TRUTH: {groundTruth}, ASSISTANT: {assistant_response}", logger)
        
        prompt = "This image gives tactile feelings of?"
        
        item = {
            
            "groundTruth": groundTruth,
            "response": assistant_response,
        }

        if not batchEval:
            evaluation = eval_fn(prompt=prompt, assistant_response=assistant_response, correct_response=groundTruth)

            score = float(getScore(evaluation))

            item["score"] = score
            item["evaluation"] = evaluation

            score_list.append(score)

            printLog(f"Evaluation output: \n {evaluation}", logger)

        scores[i] = item

        
        printLog("*"*100+"\n", logger)
    
    if save or batchEval:
        with open(filePath, 'w', encoding='utf-8') as json_file:
            json.dump(scores, json_file, ensure_ascii=False, indent=4)

    if not batchEval:
        result = getSummary(score_list)
        printLog(f"Reult: \n\t{result}.", logger)
    else:
        conductBatchInputFile(model.name, config.EVAL_PROXY_MODEL)



if __name__ == "__main__":
    configPath = "/home/aa/Desktop/WJL/VTRAG/Config/Config.yaml"

    config = getConfigFromYaml(configPath)
    model = SyncVote(config.Model.args, config.mission)
    TestOnSSVTDataset(model=model, batchEval=True, save=True, reTest=True)
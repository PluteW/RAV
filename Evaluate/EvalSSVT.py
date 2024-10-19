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
from Dataloader.SSVTPDS import SSVTPDS
from Model.SyncVote import SyncVote
from utils.EvalFunctions import (
    EVAL_PROMPT,
    get_LLama_evaluator,
    get_Proxy_evaluator,
    get_QWen_evaluator,
)
from utils.Logger import logger, printLog
from utils.Path import EVAL_RESULT_PATH, MISSION_DESCRIPTION
from utils.tools import getScore, getSummary, setSeed

basicConfigPath = "/home/aa/Desktop/WJL/VTRAG/Config/Config.yaml"

def TestOnSSVTDataset(
        model: object=None,
        dataloader:DataLoader=None,
        eval_fn:callable=None,
        configPath:str=basicConfigPath, 
        mission:str=MISSION_DESCRIPTION, 
        save=False,
        reTest=False,
    ):

    filePath = f"{EVAL_RESULT_PATH}/{model.name}.json"

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

    if eval_fn == None:
        config = getConfigFromYaml(configPath)
        # eval_fn = get_QWen_evaluator(config, EVAL_PROMPT)
        # eval_fn = get_LLama_evaluator(config, EVAL_PROMPT)
        eval_fn = get_Proxy_evaluator(config, EVAL_PROMPT)

    scores = {}
    score_list = []
    
    for i, item in zip(range(len(datasets)),dataloader):
        visionPath = item[0][0]
        touchPath = item[1][0]
        vision = Image.open(visionPath)
        touch = Image.open(touchPath)

        assistant_response = model.answer(vision, touch)
        groundTruth = item[2][0].strip().replace(" ","")

        prompt = "This image gives tactile feelings of?"
        evaluation = eval_fn(prompt=prompt, assistant_response=assistant_response, correct_response=groundTruth)

        score = float(getScore(evaluation))
        scores[i] = {
            "score": score,
            "groundTruth": groundTruth,
            "response": assistant_response,
            "evaluation": evaluation,
        }
        score_list.append(score)

        printLog(f"Item index {i}\n\tGROUND TRUTH: {groundTruth}, ASSISTANT: {assistant_response}", logger)
        printLog(f"Evaluation output: \n {evaluation}", logger)
        printLog("*"*100+"\n", logger)
    
    result = getSummary(score_list)
    printLog(f"Reult: \n\t{result}.", logger)

    if save:
        with open(filePath, 'w', encoding='utf-8') as json_file:
            json.dump(scores, json_file, ensure_ascii=False, indent=4)

    

if __name__ == "__main__":
    configPath = "/home/aa/Desktop/WJL/VTRAG/Config/Config.yaml"

    config = getConfigFromYaml(configPath)
    model = SyncVote(config.Model.args, config.mission)
    TestOnSSVTDataset(model=model, save=True, reTest=True)
import json
import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import logging

from Config.Config import getConfigFromYaml
from utils.EvalFunctions import EVAL_PROMPT, SYSTEM_PROMPT
from utils.Logger import logger, printLog
from utils.Path import EVAL_RESULT_PATH
from utils.tools import getJson, getScore, getSummary, saveJson


def conductBatchInputFile(dataset, model, gptModel, checkId=False):
    filePath = f"{EVAL_RESULT_PATH}/{dataset}/{model}.json"
    loaded_scores = getJson(filePath)
    
    if checkId:
        checkIds(dataset, model, gptModel)

    fileOutPath = f"{EVAL_RESULT_PATH}/{dataset}/{model}-{gptModel}-BatchInput.jsonl"

    with open(fileOutPath, 'w', encoding='utf-8') as jsonl_file:
        for key, value in loaded_scores.items():

            groundTruth = value["groundTruth"]
            assistant_response = value["response"]
            prompt = "This image gives tactile feelings of?"

            evalprompt = EVAL_PROMPT.format(prompt=prompt, assistant_response=assistant_response, correct_response=groundTruth)

            json_entry = {
                "custom_id": key,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": gptModel,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": evalprompt}
                    ],
                    # "max_tokens": 1000
                }
            }

            printLog(f"Query for Item {key} Ready!", logger)

            jsonl_file.write(json.dumps(json_entry, ensure_ascii=False) + '\n')
        printLog(f"Query for Items Success!", logger)

def formatBatchOutputFile(dataset, model, gptModel):
    filePath = f"{EVAL_RESULT_PATH}/{dataset}/{model}.json"
    loaded_scores = getJson(filePath)
    
    fileInputPath = f"{EVAL_RESULT_PATH}/{dataset}/{model}-{gptModel}-BatchOutput.json"

    batchOutput = getJson(fileInputPath)
    
    filtered_data = {}
    score_list = []

    for entry in batchOutput:
        custom_id = entry.get("custom_id")
        response_content = entry.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")

        status_code = entry.get("response", {}).get("status_code", 0)

        score = float(getScore(response_content))
        score_list.append(score)

        if status_code == 200:

            printLog(f"Query for Item {custom_id} Success!", logger)
            filtered_data[custom_id] = {
                "score": score,
                "groundTruth": loaded_scores[custom_id]["groundTruth"],
                "response": loaded_scores[custom_id]["response"],
                "model": gptModel,
                "evaluation": response_content
            }
        else:
            printLog(f"Query for Item {custom_id} Failed!", logger, logging.WARNING)
            # 特殊处理
            filtered_data[custom_id] = {
                "content": "Error: Status code not 200"
            }

    result = getSummary(score_list)
    printLog(f"Reult: \n\t{result}.", logger)
    
    fileOutputPath = f"{EVAL_RESULT_PATH}/{dataset}/{model}-{gptModel}-BatchFormatOutput.json"

    saveJson(fileOutputPath, filtered_data)
    # with open(fileOutputPath, 'w', encoding='utf-8') as jsonl_outfile:
    #     json.dump(filtered_data, jsonl_outfile, ensure_ascii=False, indent=4)

def checkFileIDs(fpFrom, fp):
    itemsBase = getJson(fp)
    itemsFrom = getJson(fpFrom)

    items = {}
    for k in itemsBase:
        items[k] = itemsFrom[k]

    fp_ = fpFrom.split(".")[0]+"-Checked.json"
    if len(items.items()) != len(itemsFrom.items()):
        saveJson(fp_, items)

def checkIds(dataset, model, gptModel):
    # gpt4 output
    filePath = f"{EVAL_RESULT_PATH}/{dataset}/GPT4V.json"
    
    fileBasePath = f"{EVAL_RESULT_PATH}/{dataset}/{model}.json"
    if os.path.exists(fileBasePath):
        checkFileIDs(fileBasePath, filePath)

    # fileInputPath = f"{EVAL_RESULT_PATH}/{dataset}/{model}-{gptModel}-BatchInput.jsonl"
    # if os.path.exists(fileInputPath):
    #     checkFileIDs(fileInputPath, filePath)
        
    # fileOutputPath = f"{EVAL_RESULT_PATH}/{dataset}/{model}-{gptModel}-BatchOutput.json"
    # if os.path.exists(fileOutputPath):
    #     checkFileIDs(fileOutputPath, filePath)

    fileFMOutputPath = f"{EVAL_RESULT_PATH}/{dataset}/{model}-{gptModel}-BatchFormatOutput.json"
    if os.path.exists(fileFMOutputPath):
        checkFileIDs(fileFMOutputPath, filePath)
    

if __name__ == "__main__":
    configPath = "/home/aa/Desktop/WJL/VTRAG/Config/Config.yaml"

    config = getConfigFromYaml(configPath)

    # conductBatchInputFile(config.dataset, config.Model.name, config.EVAL_PROXY_MODEL)

    formatBatchOutputFile(config.dataset, config.Model.name, config.EVAL_PROXY_MODEL)

    # checkIds(config.dataset, config.Model.name, config.EVAL_PROXY_MODEL)

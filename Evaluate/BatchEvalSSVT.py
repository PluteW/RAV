import json
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import logging

from Config.Config import getConfigFromYaml
from utils.EvalFunctions import EVAL_PROMPT, SYSTEM_PROMPT
from utils.Logger import logger, printLog
from utils.Path import EVAL_RESULT_PATH
from utils.tools import getScore, getSummary


def conductBatchInputFile(dataset, model, gptModel):
    filePath = f"{EVAL_RESULT_PATH}/{dataset}/{model}.json"
    with open(filePath, 'r', encoding='utf-8') as json_file:
        loaded_scores = json.load(json_file)

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
    with open(filePath, 'r', encoding='utf-8') as json_file:
        loaded_scores = json.load(json_file)
    
    fileInputPath = f"{EVAL_RESULT_PATH}/{dataset}/{model}-{gptModel}-BatchOutput.json"

    with open(fileInputPath, 'r', encoding='utf-8') as json_file:
        batchOutput = json.load(json_file)
    
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

    with open(fileOutputPath, 'w', encoding='utf-8') as jsonl_outfile:
        json.dump(filtered_data, jsonl_outfile, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    configPath = "/home/aa/Desktop/WJL/VTRAG/Config/Config.yaml"

    config = getConfigFromYaml(configPath)

    # conductBatchInputFile(config.dataset, config.Model.name, config.EVAL_PROXY_MODEL)

    formatBatchOutputFile(config.dataset, config.Model.name, config.EVAL_PROXY_MODEL)

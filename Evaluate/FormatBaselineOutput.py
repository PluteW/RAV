import json
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import logging
import time

from Config.Config import getConfigFromYaml
from utils.EvalFunctions import EVAL_PROMPT, SYSTEM_PROMPT
from utils.Logger import logger, printLog
from utils.Path import EVAL_RESULT_PATH
from utils.tools import getScore, getSummary


def formatBatchOutputFile(dataset, model, gptModel, fileName):
    # 参考的 ID 文件
    # filePath = f"{EVAL_RESULT_PATH}/{model}.json"
    filePath = f"/home/aa/Desktop/WJL/VTRAG/Evaluate/EvalResult/{dataset}/SyncVote.json"
    with open(filePath, 'r', encoding='utf-8') as json_file:
        loaded_scores = json.load(json_file)
    
    # 目标的规整文件
    # fileInputPath = f"{EVAL_RESULT_PATH}/{model}-{gptModel}-BatchOutput.json"
    fileInputPath = f"/home/aa/Desktop/WJL/VTRAG/Evaluate/EvalResult/baseline/{dataset}/{fileName}.json"
    with open(fileInputPath, 'r', encoding='utf-8') as json_file:
        batchOutput = json.load(json_file)
    
    filtered_data = {}
    score_list = []

    response_data = {}

    for idx, item in enumerate(batchOutput):
        label = item["label"].strip().replace(" ","")
        score = float(getScore(item["evaluation"]))
        score_list.append(score)
        for key, value in loaded_scores.items():
            groundTruth = value["groundTruth"]
            
            # 匹配
            if label == groundTruth:  # 这里可以根据需求调整匹配条件
                if label in filtered_data.keys():
                    Warning("Duplicate labels! ")
                    printLog("Duplicate labels!.", logger, logging.WARNING)
                    time.sleep(5)

                printLog(f"Got it with label: \t{label}.", logger)
                filtered_data[key] = {
                    "score": score,
                    "groundTruth": groundTruth,
                    "response": item["generated response"],
                    "model": gptModel,
                    "evaluation": item["evaluation"],
                    "image_fp": item["image_fp"],
                    "tactile_fp": item["tactile_fp"]
                }

                response_data[key] = {
                    "groundTruth": groundTruth,
                    "response": item["generated response"],
                }

                break  # 找到匹配后退出内层循环
    filtered_data = dict(sorted(filtered_data.items(), key=lambda item: int(item[0]), reverse=False))

    response_data = dict(sorted(response_data.items(), key=lambda item: int(item[0]), reverse=False))

    result = getSummary(score_list)
    printLog(f"Reult: \n\t{result}.", logger)

    respone_file = f"{EVAL_RESULT_PATH}/{dataset}/{model}.json"
    with open(respone_file, 'w', encoding='utf-8') as jsonl_outfile:
        json.dump(filtered_data, jsonl_outfile, ensure_ascii=False, indent=4)


    # fileOutputPath = f"{EVAL_RESULT_PATH}/{dataset}/{model}-{gptModel}-FormatOutput.json"
    # with open(fileOutputPath, 'w', encoding='utf-8') as jsonl_outfile:
    #     json.dump(filtered_data, jsonl_outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    formatBatchOutputFile("HCT", "GPT4V", "gpt4", "GPT4V-gpt-4v-preview-Output")

    # formatBatchOutputFile("HCT", "TVL-B", "gpt4", "tvl_llama_vitb")

    # formatBatchOutputFile("HCT", "TVL-BGS", "gpt4", "tvl_llama_vits_bgs")

    # formatBatchOutputFile("HCT", "TVL-S", "gpt4", "tvl_llama_vits")

    # formatBatchOutputFile("HCT", "TVL-TINY", "gpt4", "tvl_llama_vittiny")
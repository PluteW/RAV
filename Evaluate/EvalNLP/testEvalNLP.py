import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

import json
from statistics import mean, stdev

from Evaluate.EvalNLP.BLEU import compare_similarity as bleu_similarity
from Evaluate.EvalNLP.CIDEr import compare_similarity as cider_similarity
from Evaluate.EvalNLP.METEOR import compare_similarity as meteor_similarity
from Evaluate.EvalNLP.ROUGE import compare_similarity as rouge_similarity


def read_and_compare(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    ground_truths = [value['groundTruth'].split(',') for value in data.values()]
    responses = [value['response'].split(',') for value in data.values()]
    bleu_similarity_result = bleu_similarity(ground_truths, responses)
    cider_similarity_result = cider_similarity(ground_truths, responses)
    meteor_similarity_result = meteor_similarity(ground_truths, responses)
    rouge_similarity_result = rouge_similarity(ground_truths, responses)

    similarities = {
        'BLEU': mean(bleu_similarity_result),
        'CIDEr': mean(cider_similarity_result),
        'METEOR': mean(meteor_similarity_result),
        'ROUGE': mean(rouge_similarity_result),
        'BLEUstdev': stdev(bleu_similarity_result),
        'CIDErstdev': stdev(cider_similarity_result),
        'METEORstdev': stdev(meteor_similarity_result),
        'ROUGEstdev': stdev(rouge_similarity_result)
    }
    
    return similarities

if __name__ == "__main__":
    file_path = "/home/aa/Desktop/WJL/VTRAG/Evaluate/EvalResult/TVL/TVL-TINY.json"
    results = read_and_compare(file_path)
    print(results)
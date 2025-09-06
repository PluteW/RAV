import nltk
from nltk.translate.bleu_score import corpus_bleu


def compare_similarity(ground_truths, responses):
    return [corpus_bleu([gt], [resp],weights=(1,0,0,0)) for gt, resp in zip(ground_truths, responses)]
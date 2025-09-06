from rouge_score import rouge_scorer


def compare_similarity(ground_truths, responses):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return [scorer.score(','.join(gt), ','.join(resp))['rougeL'].fmeasure for gt, resp in zip(ground_truths, responses)]
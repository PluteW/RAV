from pycocoevalcap.cider.cider import Cider


def compare_similarity(ground_truths, responses):
    cider = Cider()
    gts = {i: [','.join(gt)] for i, gt in enumerate(ground_truths)}
    res = {i: [','.join(resp)] for i, resp in enumerate(responses)}
    score, _ = cider.compute_score(gts, res)
    return [score] * len(ground_truths)  # CIDEr returns a single score for the corpus
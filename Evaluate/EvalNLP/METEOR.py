from nltk import download
from nltk.translate.meteor_score import meteor_score

# download('wordnet')
# download('omw-1.4')

def compare_similarity(ground_truths, responses):
    return [meteor_score([gt], resp) for gt, resp in zip(ground_truths, responses)]
# , alpha=0.8, beta=0.5
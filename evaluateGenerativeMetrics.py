# -- METRICS --
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk

def evaluate_generative_metrics(tatbestand, retrieved_document):

    # 1) Tokenize reference & hypothesis
    tatbestand_tokens = nltk.word_tokenize(tatbestand)
    retrieved_tokens  = nltk.word_tokenize(retrieved_document)
    
    # 2) BLEU Score with smoothing
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([tatbestand_tokens], retrieved_tokens, smoothing_function=smooth)

    # 3) ROUGE Scores
    # For rouge_score, you can pass strings (like below)
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = rouge.score(tatbestand, retrieved_document)
  
    # 4) METEOR Score
    # meteor_score expects a list of token-lists for references, and a token-list for hypothesis.
    # So we do:
    meteor = meteor_score(
        [tatbestand_tokens],   # list of *lists* of tokens (references)
        retrieved_tokens       # single list of tokens (hypothesis)
    )

    return {
        "BLEU": bleu,
        "ROUGE-1": rouge_scores["rouge1"].fmeasure,
        "ROUGE-2": rouge_scores["rouge2"].fmeasure,
        "ROUGE-L": rouge_scores["rougeL"].fmeasure,
        "METEOR": meteor,
    }
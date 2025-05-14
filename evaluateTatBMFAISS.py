import os
import json
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np

import nltk
nltk.download('wordnet')  # for METEOR
nltk.download('punkt_tab')
from rank_bm25 import BM25Okapi
from collections import defaultdict
from typing import List, Dict, Tuple

# -- METRICS --
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import tiktoken

# ----- 1) Load environment, etc. -----

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ----- 2) Utility functions -----

def count_tokens(text, model="text-embedding-3-small"):
    """
    Count the number of tokens in a given text using a specified model.
    """
    tokenizer = tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(text))
    import numpy as np

def convert_to_serializable(obj):
    """
    Recursively convert objects to JSON-serializable types.
    """
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


def evaluate_generative_metrics(tatbestand, retrieved_document):
    """
    Evaluate a single retrieved document against the Tatbestand using generative metrics.
    """

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
    # Alternatively, if you want to pass tokenized text, you'd need to join tokens back into strings,
    # e.g. rouge.score(" ".join(tatbestand_tokens), " ".join(retrieved_tokens))

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
def compute_metrics(relevant_cases, retrieved_ids):
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_cases)

    tp = len(retrieved_set & relevant_set)  # True positives
    fp = len(retrieved_set - relevant_set)  # False positives
    fn = len(relevant_set - retrieved_set)  # False negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    coverage = len(relevant_set & retrieved_set) / len(relevant_set) if relevant_set else 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Coverage": coverage
    }
# ----- 3) Build the BM25 index from your corpus -----

# We'll assume "filtered_cases_90_percent.json" is your corpus with 'id' and 'structured_content.tatbestand'.
with open("training_cases.json", "r", encoding="utf-8") as f:
    corpus_cases = json.load(f)

bm25_corpus = []  # list of token lists
bm25_docs = []    # parallel list of doc metadata + text

for c in tqdm(corpus_cases, desc="Building Corpus"):
    doc_id = c.get("id")
    tatbestand_text = c.get("structured_content", {}).get("tatbestand", "").strip()
    if not tatbestand_text:
        continue

    # Tokenize for BM25
    tokens = tatbestand_text.split()

    bm25_corpus.append(tokens)
    bm25_docs.append({
        "id": doc_id,
        "text": tatbestand_text
    })

bm25_index = BM25Okapi(bm25_corpus)

def bm25_search_with_score(query_text: str, k=10) -> List[Tuple[Document, float]]:
    """
    Mimics 'similarity_search_with_score' but using BM25.
    Returns a list of (Document, score) tuples.
    """
    query_tokens = query_text.split()
    scores = bm25_index.get_scores(query_tokens)

    # Sort docs by descending score
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # Build the same shape as FAISS: list[(Document, float)]
    results = []
    for idx in sorted_indices[:k]:
        doc_meta = bm25_docs[idx]
        doc_obj = Document(
            page_content=doc_meta["text"],
            metadata={"id": doc_meta["id"]}
        )
        # BM25 gives bigger = better, while FAISS is smaller = better distance
        # But we can just store the BM25 raw score
        results.append((doc_obj, float(scores[idx])))

    return results

# ----- 4) Load FAISS index -----

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Specify the correct model used during FAISS creation
    openai_api_key=openai_api_key
)
vectorstore = FAISS.load_local(
    "faiss_legal_index_tatbestand", 
    embeddings,
    allow_dangerous_deserialization=True
)

# ----- 5) Compare on test cases from "matching_cases.json" -----

with open('enriched_cases.json', 'r', encoding='utf-8') as f:
    test_cases = json.load(f)

test_cases = test_cases[:100]  # limit for demonstration

all_metrics = []
rank_metrics_faiss = defaultdict(lambda: defaultdict(list))
rank_metrics_bm25 = defaultdict(lambda: defaultdict(list))
# Initialize storage for match statistics
relevant_case_stats = {"FAISS": defaultdict(int), "BM25": defaultdict(int)}

for case in tqdm(test_cases, desc="Processing Documents"):
    tatbestand = case["structured_content"].get("tatbestand", "").strip()
    case_id = case.get("id")
    relevant_cases = case.get("relevant_cases", [])
    # if not relevant_cases:
    #     continue
    if not tatbestand:
        print(f"Skipping case {case_id}: empty tatbestand.")
        continue

    token_count = count_tokens(tatbestand, model="text-embedding-3-small")
    if token_count > 8000:
        print(f"Skipping case {case_id}: tatbestand exceeds 8000 tokens ({token_count}).")
        continue
    # Initialize ranks for relevant cases
    relevant_case_ranks = {"FAISS": {}, "BM25": {}}
    # -- A) FAISS retrieval
    faiss_results = vectorstore.similarity_search_with_score(tatbestand, k=10)
    faiss_ids = {doc.metadata.get("id") for doc, _ in faiss_results}

    # -- B) BM25 retrieval
    bm25_results = bm25_search_with_score(tatbestand, k=10)
    bm25_ids = {doc.metadata.get("id") for doc, _ in bm25_results}


    # Evaluate rank at which relevant cases are found
    for method, results in [("FAISS", faiss_results), ("BM25", bm25_results)]:
        for rank, (doc, _) in enumerate(results, start=1):
            retrieved_id = doc.metadata.get("id")
            if retrieved_id in relevant_cases:
                relevant_case_ranks[method][retrieved_id] = rank
    #check Accuracy for both
    #are there relevant cases?
    # Check accuracy for both retrieval methods
    if relevant_cases:
        for relevant_case in relevant_cases:
            # Check BM25 results
            if relevant_case in bm25_ids:
                relevant_case_stats["BM25"]["matches"] += 1
            else:
                relevant_case_stats["BM25"]["misses"] += 1

            # Check FAISS results
            if relevant_case in faiss_ids:
                relevant_case_stats["FAISS"]["matches"] += 1
            else:
                relevant_case_stats["FAISS"]["misses"] += 1
    faiss_metrics_for_case = []
    for rank, (doc, score) in enumerate(faiss_results, start=1):
        retrieved_content = doc.page_content
        metrics = evaluate_generative_metrics(tatbestand, retrieved_content)
        metrics.update({
            "Retrieved ID": doc.metadata.get("id", f"retrieved_doc_{rank}"),
            "Similarity Score (FAISS)": score,  
            "Rank": rank
        })
        faiss_metrics_for_case.append(metrics)
        # Store for averaging by rank
        for key, val in metrics.items():
            if key not in ["Retrieved ID", "Rank", "Similarity Score (FAISS)"]:
                rank_metrics_faiss[rank][key].append(val)

    # Evaluate BM25 retrieval
    bm25_metrics_for_case = []
    for rank, (doc, score) in enumerate(bm25_results, start=1):
        retrieved_content = doc.page_content
        metrics = evaluate_generative_metrics(tatbestand, retrieved_content)
        metrics.update({
            "Retrieved ID": doc.metadata.get("id", f"retrieved_doc_{rank}"),
            "BM25 Score": score,
            "Rank": rank
        })
        bm25_metrics_for_case.append(metrics)
        # Store for averaging by rank
        for key, val in metrics.items():
            if key not in ["Retrieved ID", "Rank", "BM25 Score"]:
                rank_metrics_bm25[rank][key].append(val)

    # Combine into a final dict
    combined_case_metrics = {
        "Case ID": case_id,
        "Tatbestand Tokens": token_count,
        "FAISS Retrievals": faiss_metrics_for_case,
        "BM25 Retrievals": bm25_metrics_for_case
    }

    all_metrics.append(combined_case_metrics)

# ----- 6) Compute average scores by rank for FAISS vs BM25 -----

faiss_rank_averages = {}
for rank, metric_dict in rank_metrics_faiss.items():
    avg = {}
    for metric_name, values in metric_dict.items():
        avg[metric_name] = sum(values) / len(values) if values else 0.0
    faiss_rank_averages[rank] = avg

bm25_rank_averages = {}
for rank, metric_dict in rank_metrics_bm25.items():
    avg = {}
    for metric_name, values in metric_dict.items():
        avg[metric_name] = sum(values) / len(values) if values else 0.0
    bm25_rank_averages[rank] = avg

# ----- 7) Save or print results -----

# Convert data to JSON serializable format
serializable_metrics = convert_to_serializable(all_metrics)

# Save all per-case metrics
with open("bm25_vs_faiss_metrics.json", "w", encoding="utf-8") as f:
    json.dump(serializable_metrics, f, indent=4)

print("\n===== Relevant cases ranks =====")
print(relevant_case_ranks)
print("\n===== AVERAGE SCORES BY RANK (FAISS) =====")
for rank in sorted(faiss_rank_averages.keys()):
    print(f"Rank {rank} => {faiss_rank_averages[rank]}")

print("\n===== AVERAGE SCORES BY RANK (BM25) =====")
for rank in sorted(bm25_rank_averages.keys()):
    print(f"Rank {rank} => {bm25_rank_averages[rank]}")

# Summarize results
print("\n===== RELEVANT CASE MATCH SUMMARY =====")
print("BM25:")
print(f"  Matches: {relevant_case_stats['BM25']['matches']}")
print(f"  Misses: {relevant_case_stats['BM25']['misses']}")

print("\nFAISS:")
print(f"  Matches: {relevant_case_stats['FAISS']['matches']}")
print(f"  Misses: {relevant_case_stats['FAISS']['misses']}")

print("\nSide-by-side BM25 vs. FAISS evaluation completed. Results saved to 'bm25_vs_faiss_metrics.json'.")

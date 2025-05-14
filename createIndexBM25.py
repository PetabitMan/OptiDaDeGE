import os
import json
from dotenv import load_dotenv
from typing import List, Dict
from rank_bm25 import BM25Okapi

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def createBM25Index():
    # A container for storing our documents and their metadata
    documents = []
    metadata = []

    # Tokenized corpus for BM25
    bm25_corpus = []

    # Suppose we load from "filtered_cases_90_percent.json" or any source
    with open("filtered_cases_90_percent.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    for case in cases:
        case_id = case.get("id")
        tatbestand_text = case["structured_content"].get("tatbestand", "").strip()
        
        # Skip if there's no tatbestand text
        if not tatbestand_text:
            continue
        
        # Keep track of the full text in a list
        documents.append(tatbestand_text)
        
        # Save metadata (like ID). We'll align indexes.
        metadata.append({"id": case_id})
        
        # For BM25, we need a tokenized version of the text
        # Simple approach: split on whitespace
        tokenized_text = tatbestand_text.split()
        bm25_corpus.append(tokenized_text)

    # Initialize BM25
    bm25_index = BM25Okapi(bm25_corpus)
    return bm25_index, documents, metadata


def bm25_search(query_text: str, index, documents: List[str], metadata: List[Dict], k=5) -> List[Dict]:
    """
    Searches the BM25 index for the top k matching documents.
    Returns a list of dicts with 'text' and 'metadata'.
    """
    query_tokens = query_text.split()  # simple whitespace split
    scores = index.get_scores(query_tokens)
    
    # Sort the documents in descending order of relevance
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    top_matches = []
    for idx in sorted_indices[:k]:
        top_matches.append({
            "text": documents[idx],
            "metadata": metadata[idx],
            "score": scores[idx]
        })
    return top_matches


# Create BM25 index and load documents/metadata
bm25_index, documents, metadata = createBM25Index()

# Test query
test_tat = """Tatbestand: 2Der Kläger begehrt die Erstattung von 740 Euro nebst Zinsen für Aufwendungen, die er nach eigenen Angaben hatte wegen der Teilnahme an einem Weiterbildungskurs und für Fahrstunden der Klasse 2. ..."""
results = bm25_search(test_tat, bm25_index, documents, metadata, k=5)

# Print results
for i, r in enumerate(results, start=1):
    print(f"Match {i}: (Score: {r['score']:.2f})")
    print(f"ID: {r['metadata']['id']}")
    print(f"Text (truncated): {r['text'][:300]}...")
    print()

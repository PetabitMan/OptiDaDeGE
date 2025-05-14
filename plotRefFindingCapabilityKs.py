import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Set
from tqdm import tqdm

import matplotlib.pyplot as plt
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------
KS = [1, 2, 3, 5, 7, 10]
K_MAX = max(KS)
# Color scheme
COLOR_PERCENT = "#002147"   # Midnight Blue for percentage line

COLOR_TOTAL = "#4682B4"      # Steel Blue
COLOR_OVERLAP = "#B0C4DE"    # Light Blue for overlap
GRID_COLOR = "#D3D3D3"       # Light Gray

# ------------------------------------------------------------------
# Load environment variables & initialize vector store
# ------------------------------------------------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key
)
vectorstore = FAISS.load_local(
    "faiss_legal_index_tatbestand",
    embeddings,
    allow_dangerous_deserialization=True
)

# ------------------------------------------------------------------
# Helper: similarity search
# ------------------------------------------------------------------

def find_similar_case_docs(tatbestand: str, jurisdiction: str, k: int):
    candidates = vectorstore.similarity_search(tatbestand, k=k * 3)
    if jurisdiction:
        filtered = [
            doc for doc in candidates
            if doc.metadata and str(doc.metadata.get("jurisdiction", "")).lower() == jurisdiction.lower()
        ]
    else:
        filtered = candidates
    return filtered[:k]

# ------------------------------------------------------------------
# Load enriched cases & training cases
# ------------------------------------------------------------------
print("Loading Data...")
with open("enriched_cases.json", "r", encoding="utf-8") as f:
    enriched_cases = json.load(f)
with open("training_cases.json", "r", encoding="utf-8") as f:
    training_list = json.load(f)
# Fast lookup by case ID
training_cases = {case["id"]: case for case in training_list if "id" in case}
print("Finished Data Loading!")
# ------------------------------------------------------------------
# Precompute top K_MAX similar case IDs and query case refs
# ------------------------------------------------------------------
similar_ids_map: Dict[str, List[str]] = {}
query_refs_map: Dict[str, Set[str]] = {}
num_cases = 0

for case in tqdm(enriched_cases[:100], desc="Finding Cases"):
    case_id = case.get("id")
    tatbestand = case.get("structured_content", {}).get("tatbestand", "")
    if not case_id or not tatbestand:
        continue
    jurisdiction = case.get("court", {}).get("jurisdiction", "")
    # Compute similar IDs once for maximum k
    docs_max = find_similar_case_docs(tatbestand, jurisdiction, K_MAX)
    similar_ids_map[case_id] = [doc.metadata.get("id") for doc in docs_max if doc.metadata.get("id")]
    # Store simple_refs of the query (enriched) case
    query_refs_map[case_id] = set(case.get("simple_refs", []))

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Compute average percentages for overlap and total found relative to query refs
# ------------------------------------------------------------------
avg_percent_found: List[float] = []  # overlap relative to query refs
avg_percent_total: List[float] = []  # total found relative to query refs

for k in KS:
    total_percent = 0.0
    found_percent = 0.0
    valid_cases = 0

    for case_id, candidate_ids in similar_ids_map.items():
        query_refs = query_refs_map.get(case_id, set())
        if not query_refs:
            continue
        # Collect refs from top-k similar cases
        refs: Set[str] = set()
        for cid in candidate_ids[:k]:
            tc = training_cases.get(cid)
            if tc:
                refs.update(tc.get("simple_refs", []))
        # Compute percentages
        overlap = len(query_refs & refs)
        percent_overlap = (overlap / len(query_refs)) * 100
        percent_total = (len(refs) / len(query_refs)) * 100

        found_percent += percent_overlap
        total_percent += percent_total
        valid_cases += 1

    avg_percent_found.append(found_percent / valid_cases if valid_cases else 0)
    avg_percent_total.append(total_percent / valid_cases if valid_cases else 0)

# ------------------------------------------------------------------
# Plotting average percentages
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5), facecolor="#FFFFFF")
plt.plot(KS, avg_percent_found, marker='o', label='Ø % überlappender Normen', color=COLOR_PERCENT)
plt.plot(KS, avg_percent_total, marker='s', label='Ø % gefundener Normen insgesamt', color="#4682B4")

plt.xlabel('k (Anzahl ähnlicher Dokumente)')
plt.ylabel('Durchschnittlicher Prozentsatz (%) bezogen auf Query-Refs')
plt.title('Durchschnittlicher Anteil der gefundenen und überlappenden Normen pro Fall')
plt.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.5)
plt.legend()

# Save and show
output_file = 'avg_norm_percent.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor="#FFFFFF")
print(f"Diagramm gespeichert als {output_file}")
plt.show()

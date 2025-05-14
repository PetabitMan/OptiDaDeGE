import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Farbschema
COLORS = {
    "primary": {
        "midnight_blue": "#002147",
        "steel_blue": "#4682B4"
    },
    "secondary": {
        "light_blue": "#B0C4DE",
        "light_gray": "#D3D3D3"
    },
    "accent": {
        "dark_gray": "#A9A9A9",
        "white": "#FFFFFF"
    }
}

# Datei mit den gefilterten Fällen (gespeicherte Gerichtsurteile)
filtered_cases_file = "sufficient_cases.json"

# Datei mit den analysierten Fällen (z. B. FAISS und BM25 Retrievals)
retrieved_cases_file = "bm25_vs_faiss_metrics.json"

print("Laden der gefilterten Fälle")
with open(filtered_cases_file, "r", encoding="utf-8") as f:
    sufficient_cases = json.load(f)

print("Laden der abgerufenen Fälle")
with open(retrieved_cases_file, "r", encoding="utf-8") as f:
    retrieved_cases_data = json.load(f)

# Funktion zum Finden eines Falls anhand seiner ID in sufficient_cases
def find_case_by_id(case_id, cases):
    for case in cases:
        if case.get("id") == case_id:
            return case
    return None

# Funktion zur Analyse der Referenzüberschneidung für beide Retrieval-Methoden
def analyze_references(case_data, retrieval_type, jurisdictionFilter):
    results = []
    
    for case in tqdm(case_data, desc=f"Processing {retrieval_type}"):
        case_id = case["Case ID"]
        retrieved_cases = case[retrieval_type]  # Zugriff auf BM25 oder FAISS
        
        # Hole Referenzen des Ausgangsfalls
        original_case = find_case_by_id(case_id, sufficient_cases)
        if not original_case:
            continue
        
        original_simple_refs = set(original_case.get("simple_refs", []))
        original_case_refs = set(original_case.get("case_references", []))
        
        for retrieved in retrieved_cases:
            retrieved_id = retrieved["Retrieved ID"]
            rank = retrieved["Rank"]
            retrieved_case = find_case_by_id(retrieved_id, sufficient_cases)
            if not retrieved_case:
                continue
            if jurisdictionFilter:
                if retrieved_case.get("court", {}).get("jurisdiction", "") != original_case.get("court", {}).get("jurisdiction", ""):
                    continue

            retrieved_simple_refs = set(retrieved_case.get("simple_refs", []))
            retrieved_case_refs = set(retrieved_case.get("case_references", []))
            
            # Berechnung der Überschneidungen
            common_simple_refs = original_simple_refs.intersection(retrieved_simple_refs)
            common_case_refs = original_case_refs.intersection(retrieved_case_refs)
            
            results.append({
                "Rank": rank,
                "Common Simple Refs": len(common_simple_refs),
                "Common Case Refs": len(common_case_refs),
                "Method": retrieval_type
            })
    return results

# Analysiere die Referenzdaten für beide Methoden
faiss_analysis = analyze_references(retrieved_cases_data, "FAISS Retrievals", False)
bm25_analysis = analyze_references(retrieved_cases_data, "BM25 Retrievals", False)

# Umwandlung in DataFrame für die Visualisierung
df = pd.DataFrame(faiss_analysis + bm25_analysis)

# Berechnung des Durchschnitts pro Rang und Methode
df_mean = df.groupby(["Rank", "Method"], as_index=False).mean()

# Speichern des Diagramms als Datei
plot_filename = "retrieval_vergleich.png"

plt.figure(figsize=(12, 6))

# FAISS Linien
sns.lineplot(
    data=df_mean[df_mean["Method"] == "FAISS Retrievals"],
    x="Rank", y="Common Simple Refs",
    marker="o", color=COLORS["primary"]["midnight_blue"], label="FAISS - Durchschnitt gemeinsame Rechtsreferenzen"
)
sns.lineplot(
    data=df_mean[df_mean["Method"] == "FAISS Retrievals"],
    x="Rank", y="Common Case Refs",
    marker="s", color=COLORS["primary"]["steel_blue"], label="FAISS - Durchschnitt gemeinsame Urteilsreferenzen"
)

# BM25 Linien
sns.lineplot(
    data=df_mean[df_mean["Method"] == "BM25 Retrievals"],
    x="Rank", y="Common Simple Refs",
    marker="o", color=COLORS["secondary"]["light_blue"], label="BM25 - Durchschnitt gemeinsame Rechtsreferenzen"
)
sns.lineplot(
    data=df_mean[df_mean["Method"] == "BM25 Retrievals"],
    x="Rank", y="Common Case Refs",
    marker="s", color=COLORS["secondary"]["light_gray"], label="BM25 - Durchschnitt gemeinsame Urteilsreferenzen"
)

plt.xlabel("Rang der abgerufenen Urteile")
plt.ylabel("Durchschnittliche Anzahl gemeinsamer Referenzen")
plt.title("Vergleich der gemeinsamen Referenzen für FAISS vs. BM25")
plt.legend()
plt.grid(True)

# Speichern der Grafik
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
plt.close()

print(f"Diagramm gespeichert als: {plot_filename}")

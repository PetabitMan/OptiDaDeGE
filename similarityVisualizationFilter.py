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

# Datei mit den analysierten Fällen (z. B. FAISS Retrievals)
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

# Funktion zur Analyse der Referenzüberschneidung für FAISS mit und ohne Jurisdiction-Filter
def analyze_references(case_data, retrieval_type, jurisdiction_filter):
    results = []
    
    for case in tqdm(case_data, desc=f"Processing {retrieval_type} - {'With' if jurisdiction_filter else 'Without'} Filter"):
        case_id = case["Case ID"]
        retrieved_cases = case[retrieval_type]  # Zugriff auf FAISS Retrievals
        
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
            if jurisdiction_filter:
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
                "Jurisdiction Filter": "With Filter" if jurisdiction_filter else "Without Filter"
            })
    return results

# Analysiere die Referenzdaten für FAISS mit und ohne Jurisdiction-Filter
faiss_no_filter = analyze_references(retrieved_cases_data, "FAISS Retrievals", False)
faiss_with_filter = analyze_references(retrieved_cases_data, "FAISS Retrievals", True)

# Umwandlung in DataFrame für die Visualisierung
df_faiss = pd.DataFrame(faiss_no_filter + faiss_with_filter)

# Berechnung des Durchschnitts pro Rang und Filterstatus
df_faiss_mean = df_faiss.groupby(["Rank", "Jurisdiction Filter"], as_index=False).mean()

# Speichern des Diagramms als Datei
plot_filename_faiss = "faiss_jurisdiction_comparison.png"

plt.figure(figsize=(12, 6))

# FAISS ohne Jurisdiction-Filter
sns.lineplot(
    data=df_faiss_mean[df_faiss_mean["Jurisdiction Filter"] == "Without Filter"],
    x="Rank", y="Common Simple Refs",
    marker="o", color=COLORS["primary"]["midnight_blue"], label="FAISS - Ohne Jurisdiction-Filter (Simple Refs)"
)
sns.lineplot(
    data=df_faiss_mean[df_faiss_mean["Jurisdiction Filter"] == "Without Filter"],
    x="Rank", y="Common Case Refs",
    marker="s", color=COLORS["primary"]["steel_blue"], label="FAISS - Ohne Jurisdiction-Filter (Case Refs)"
)

# FAISS mit Jurisdiction-Filter
sns.lineplot(
    data=df_faiss_mean[df_faiss_mean["Jurisdiction Filter"] == "With Filter"],
    x="Rank", y="Common Simple Refs",
    marker="o", color=COLORS["secondary"]["light_blue"], label="FAISS - Mit Jurisdiction-Filter (Simple Refs)"
)
sns.lineplot(
    data=df_faiss_mean[df_faiss_mean["Jurisdiction Filter"] == "With Filter"],
    x="Rank", y="Common Case Refs",
    marker="s", color=COLORS["secondary"]["light_gray"], label="FAISS - Mit Jurisdiction-Filter (Case Refs)"
)

plt.xlabel("Rang der abgerufenen Urteile")
plt.ylabel("Durchschnittliche Anzahl gemeinsamer Referenzen")
plt.title("Vergleich der FAISS Retrievals mit und ohne Jurisdiction-Filter")
plt.legend()
plt.grid(True)

# Speichern der Grafik
plt.savefig(plot_filename_faiss, dpi=300, bbox_inches="tight")
plt.close()

print(f"Diagramm gespeichert als: {plot_filename_faiss}")

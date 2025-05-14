import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Define colors according to the provided color scheme
COLORS = {
    "primary": {
        "midnight_blue": "#002147",
        "steel_blue": "#4682B4",
    },
    "secondary": {
        "light_blue": "#B0C4DE",
        "light_gray": "#D3D3D3",
    },
    "accent": {
        "dark_gray": "#A9A9A9",
        "white": "#FFFFFF",
    },
}

# File paths – adjust these if necessary
LAW_APP_RESULTS_FILE = "law_application_results.json"  # Result file from the previous program
ENRICHED_CASES_FILE = "enriched_cases.json"            # Enriched cases that contain ground-truth "simple_refs"
OUTPUT_GRAPH_FILE = "evaluation_graph.png"

def main():
    # Load law application results (each entry: {"case_id": ..., "law_applications": {ref: {Applies, does_apply}, ...}})
    with open(LAW_APP_RESULTS_FILE, "r", encoding="utf-8") as f:
        law_results = json.load(f)

    # Load enriched cases (each should contain a "simple_refs" field with the ground-truth law references)
    with open(ENRICHED_CASES_FILE, "r", encoding="utf-8") as f:
        enriched_cases = json.load(f)

    # Build a mapping from case_id to enriched case data (if available)
    enriched_cases_map = {case["id"]: case for case in enriched_cases if "id" in case}

        # --- Compute Precision, Recall, and F1 ---
    precision_ratios = []  # Per-case precision
    recall_ratios = []     # Per-case recall
    f1_ratios = []         # Per-case F1 score
    valid_case_ids = []    # For logging purposes

    # Process only the first 100 cases
    for result in tqdm(law_results[:100], desc="Evaluating Cases"):
        case_id = result.get("case_id")
        if case_id not in enriched_cases_map:
            continue

        enriched_case = enriched_cases_map[case_id]
        ground_truth_refs = enriched_case.get("simple_refs", [])
        if not ground_truth_refs:
            continue

        ground_truth_set = set(ground_truth_refs)
        # Found references from the law application results
        found_refs = set(result.get("law_applications", {}).keys())
        # System predictions: only those references with does_apply==True
        system_applied = {ref for ref in found_refs if result["law_applications"][ref].get("does_apply") is True}
        # True positives: system predictions that are in the ground truth
        true_positives = ground_truth_set.intersection(system_applied)

        # Compute precision: fraction of predicted refs that are correct
        precision = len(true_positives) / len(system_applied) if system_applied else 0
        # Compute recall: fraction of ground-truth refs that were predicted
        recall = len(true_positives) / len(ground_truth_set) if ground_truth_set else 0
        # Compute F1: harmonic mean of precision and recall
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        precision_ratios.append(precision)
        recall_ratios.append(recall)
        f1_ratios.append(f1)
        valid_case_ids.append(case_id)

    # Calculate average metrics
    avg_precision = sum(precision_ratios) / len(precision_ratios) if precision_ratios else 0
    avg_recall = sum(recall_ratios) / len(recall_ratios) if recall_ratios else 0
    avg_f1 = sum(f1_ratios) / len(f1_ratios) if f1_ratios else 0

    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average F1: {avg_f1:.2f}")

    # --- Plot the results ---
    metrics = ["Precision", "Recall", "F1"]
    values = [avg_precision, avg_recall, avg_f1]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(metrics, values, color=[
        COLORS["primary"]["midnight_blue"],
        COLORS["secondary"]["light_blue"],
        COLORS["primary"]["steel_blue"]
    ])

    ax.set_ylim(0, 1)
    ax.set_ylabel("Ratio")
    ax.set_title("Evaluation Metrics for Law References")

    # Annotate each bar with its percentage value
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height*100:.1f}%', 
                    xy=(bar.get_x() + bar.get_width() / 2, height), 
                    xytext=(0, 3),  # offset text by 3 points vertically
                    textcoords="offset points", 
                    ha='center', va='bottom',
                    color=COLORS["accent"]["dark_gray"])

    plt.tight_layout()
    plt.savefig(OUTPUT_GRAPH_FILE)
    plt.show()
    print(f"Graph saved to {OUTPUT_GRAPH_FILE}")


if __name__ == "__main__":
    main()

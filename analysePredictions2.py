import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# -------------------------------------------------------------------
# Color Palette
# -------------------------------------------------------------------
PRIMARY_COLORS = {
    "midnight_blue": "#002147",
    "steel_blue": "#4682B4"
}
SECONDARY_COLORS = {
    "light_blue": "#B0C4DE",
    "light_gray": "#D3D3D3"
}
ACCENT_COLORS = {
    "dark_gray": "#A9A9A9",
    "white": "#FFFFFF"
}

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def read_json_file(file_path):
    """Reads a JSON file and returns the parsed data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading {file_path}: {e}")
        return None

def normalize_winner_name(name):
    """Normalizes winner names to standard labels."""
    mapping = {
        "Angeklagte": "Defendant",
        "Angeklagter": "Defendant",
        "Antragsgegnerin": "Defendant",
        "Verfügungsbeklagte": "Defendant",
        "Verfügungsbeklagter": "Defendant",
        "Antragsgegner": "Defendant",
        "Beigeladene": "Defendant",
        "Beklagte": "Defendant",
        "Beklagter": "Defendant",
        "Kläger": "Plaintiff",
        "Klägerin": "Plaintiff",
        "Plaintiff": "Plaintiff",
        "Defendant": "Defendant"
    }
    return mapping.get(name, name)

def build_predictions_index(predictions_data):
    """Builds an index of predictions keyed by 'case_id'."""
    predictions_index = {}
    for prediction in predictions_data:
        case_id = prediction.get("case_id")
        #case_id = prediction.get("id")

        if case_id:
            predictions_index[case_id] = prediction
    return predictions_index

def find_prediction(case, predictions_index):
    """
    Finds a matching prediction for a given case using case_id.
    Falls back to matching on 'structured_content' if needed.
    """
    case_id = case.get("id")
    if case_id and case_id in predictions_index:
        return predictions_index[case_id]
    # Fallback: try matching based on 'structured_content'
    for prediction in predictions_index.values():
        if case.get("structured_content") == prediction.get("structured_content"):
            return prediction
    return None

def calculate_percentage_difference(actual, predicted):
    """Calculates the absolute percentage difference (returns 0 if values are missing)."""
    if actual is None or predicted is None:
        return 0
    return abs(actual - predicted)

# -------------------------------------------------------------------
# Graphing Functions
# -------------------------------------------------------------------
def save_metrics_chart(metrics, output_path):
    """
    Saves a horizontal bar chart for the provided metrics.
    'metrics' should be a list of dictionaries with 'Metric' and 'Value' keys.
    """
    metrics_df = pd.DataFrame(metrics)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(metrics_df["Metric"], metrics_df["Value"],
                   color=SECONDARY_COLORS["light_blue"],
                   edgecolor=PRIMARY_COLORS["midnight_blue"])
    ax.set_xlabel("Value", color=PRIMARY_COLORS["midnight_blue"])
    ax.set_title("Model Metrics", color=PRIMARY_COLORS["midnight_blue"])
    ax.tick_params(colors=PRIMARY_COLORS["midnight_blue"])
    
    # Annotate each bar with its value
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', va='center', color=PRIMARY_COLORS["midnight_blue"])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_actual_vs_predicted_chart(actual_counts, predicted_counts, matching_counts, output_path):
    """
    Saves a grouped bar chart comparing actual vs. predicted counts.
    """
    labels = list(actual_counts.keys())
    actual_values = [actual_counts[label] for label in labels]
    predicted_values = [predicted_counts[label] for label in labels]
    matching_values = [matching_counts[label] for label in labels]

    x = range(len(labels))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([p - bar_width for p in x], actual_values, width=bar_width,
                   label="Actual", color=SECONDARY_COLORS["light_blue"])
    bars2 = ax.bar(x, predicted_values, width=bar_width,
                   label="Predicted", color=PRIMARY_COLORS["steel_blue"])
    bars3 = ax.bar([p + bar_width for p in x], matching_values, width=bar_width,
                   label="Matching", color=ACCENT_COLORS["dark_gray"])

    ax.set_xlabel("Class", color=PRIMARY_COLORS["midnight_blue"])
    ax.set_ylabel("Counts", color=PRIMARY_COLORS["midnight_blue"])
    ax.set_title("Actual vs Predicted Counts", color=PRIMARY_COLORS["midnight_blue"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=PRIMARY_COLORS["midnight_blue"])
    ax.tick_params(colors=PRIMARY_COLORS["midnight_blue"])
    ax.legend()

    # Annotate bars with their counts
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                    f'{int(height)}', ha='center', color=PRIMARY_COLORS["midnight_blue"])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# -------------------------------------------------------------------
# Evaluation and Analysis Functions
# -------------------------------------------------------------------
def evaluate_predictions(actual_data, predictions_data):
    """
    Evaluates predictions by comparing actual vs. predicted winners and costs.
    Returns:
      - metrics: a dictionary of computed metrics.
      - y_true: list of normalized actual winners.
      - y_pred: list of normalized predicted winners.
    """
    predictions_index = build_predictions_index(predictions_data)
    y_true = []
    y_pred = []
    total_cases = 0
    correct_winner_count = 0
    correct_cost_klager = 0
    correct_cost_angeklagte = 0
    total_divergence_klager = 0.0
    total_divergence_angeklagte = 0.0
    total_relevant_cases = 0
    correct_defendant_cases = 0
    correct_plaintiff_cases = 0
    clear_cases = 0

    for actual_case in actual_data:
        prediction = find_prediction(actual_case, predictions_index)
        if not prediction:
            continue

        total_cases += 1
        actual_winner = actual_case.get("winner")
        predicted_winner = prediction.get("predicted_tenor", {}).get("winner")

        if actual_winner and predicted_winner:
            norm_actual = normalize_winner_name(actual_winner)
            norm_predicted = normalize_winner_name(predicted_winner)
            y_true.append(norm_actual)
            y_pred.append(norm_predicted)
            if norm_actual == norm_predicted:
                correct_winner_count += 1
        else:
            continue  # Skip cases without valid winner info

        actual_costs = actual_case.get("costs_borne_by")
        predicted_costs = prediction.get("predicted_tenor", {}).get("costs_borne_by")
        if actual_costs and predicted_costs:
            cost_plaintiff_actual = actual_costs.get("prozentzahl_kläger")
            cost_defendant_actual = actual_costs.get("prozentzahl_angeklagte")
            cost_plaintiff_pred = predicted_costs.get("prozentzahl_kläger")
            cost_defendant_pred = predicted_costs.get("prozentzahl_angeklagte")

            if cost_plaintiff_actual == cost_plaintiff_pred:
                correct_cost_klager += 1
            if cost_defendant_actual == cost_defendant_pred:
                correct_cost_angeklagte += 1

            total_divergence_klager += calculate_percentage_difference(cost_plaintiff_actual, cost_plaintiff_pred)
            total_divergence_angeklagte += calculate_percentage_difference(cost_defendant_actual, cost_defendant_pred)

            # Analyze clear cases:
            # For Defendant: actual winner is Defendant and Plaintiff bears 100% costs.
            if normalize_winner_name(actual_winner) == "Defendant" and cost_plaintiff_actual == 100.0:
                total_relevant_cases += 1
                if normalize_winner_name(predicted_winner) == "Defendant":
                    correct_defendant_cases += 1
            # For Plaintiff: actual winner is Plaintiff and Defendant bears 100% costs.
            elif normalize_winner_name(actual_winner) == "Plaintiff" and cost_defendant_actual == 100.0:
                total_relevant_cases += 1
                if normalize_winner_name(predicted_winner) == "Plaintiff":
                    correct_plaintiff_cases += 1

            if ((normalize_winner_name(actual_winner) == "Plaintiff" and cost_defendant_actual == 100.0) or
                (normalize_winner_name(actual_winner) == "Defendant" and cost_plaintiff_actual == 100.0)):
                clear_cases += 1

    # Compute classification metrics if any cases were evaluated
    if y_true:
        accuracy = accuracy_score(y_true, y_pred) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=["Plaintiff", "Defendant"], zero_division=0
        )
        precision_overall, recall_overall, f1_overall, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    else:
        accuracy = 0
        precision = [0, 0]
        recall = [0, 0]
        f1 = [0, 0]
        precision_overall = recall_overall = f1_overall = 0

    avg_divergence_klager = total_divergence_klager / total_cases if total_cases > 0 else 0
    avg_divergence_angeklagte = total_divergence_angeklagte / total_cases if total_cases > 0 else 0

    metrics = {
        "Total Cases": total_cases,
        "Winner Accuracy (%)": accuracy,
        "Correct Winner Predictions": correct_winner_count,
        "Plaintiff Precision": precision[0] * 100,
        "Plaintiff Recall": recall[0] * 100,
        "Plaintiff F1": f1[0] * 100,
        "Defendant Precision": precision[1] * 100,
        "Defendant Recall": recall[1] * 100,
        "Defendant F1": f1[1] * 100,
        "Overall Precision": precision_overall,
        "Overall Recall": recall_overall,
        "Overall F1": f1_overall,
        "Correct Costs (Plaintiff %)": correct_cost_klager,
        "Correct Costs (Defendant %)": correct_cost_angeklagte,
        "Average Divergence (Plaintiff %)": avg_divergence_klager,
        "Average Divergence (Defendant %)": avg_divergence_angeklagte,
        "Correct Defendant Clear Cases": correct_defendant_cases,
        "Correct Plaintiff Clear Cases": correct_plaintiff_cases,
        "Total Relevant Clear Cases": total_relevant_cases,
        "Total Clear Cases": clear_cases
    }
    return metrics, y_true, y_pred

def generate_charts(metrics, y_true, y_pred, file_path):
    """Generates and saves the metrics and actual vs. predicted charts."""
    # Prepare metrics chart data
    metrics_chart_data = [
        {"Metric": "Winner Accuracy (%)", "Value": metrics["Winner Accuracy (%)"]},
        {"Metric": "Plaintiff Precision", "Value": metrics["Plaintiff Precision"]},
        {"Metric": "Plaintiff Recall", "Value": metrics["Plaintiff Recall"]},
        {"Metric": "Plaintiff F1", "Value": metrics["Plaintiff F1"]},
        {"Metric": "Defendant Precision", "Value": metrics["Defendant Precision"]},
        {"Metric": "Defendant Recall", "Value": metrics["Defendant Recall"]},
        {"Metric": "Defendant F1", "Value": metrics["Defendant F1"]}
    ]
    save_metrics_chart(metrics_chart_data, f"results/metrics_chart_{file_path}.png")

    # Prepare actual vs. predicted counts
    actual_counts = {"Plaintiff": y_true.count("Plaintiff"), "Defendant": y_true.count("Defendant")}
    predicted_counts = {"Plaintiff": y_pred.count("Plaintiff"), "Defendant": y_pred.count("Defendant")}
    matching_counts = {
        "Plaintiff": sum(1 for a, p in zip(y_true, y_pred) if a == "Plaintiff" and p == "Plaintiff"),
        "Defendant": sum(1 for a, p in zip(y_true, y_pred) if a == "Defendant" and p == "Defendant")
    }
    save_actual_vs_predicted_chart(actual_counts, predicted_counts, matching_counts, f"results/actual_vs_predicted_chart_{file_path}.png")

def display_metrics(metrics):
    """Prints out computed metrics in a clear, formatted way."""
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def main():
    # Specify file paths (adjust as needed)
    actual_file = 'enriched_cases.json'
    # fileNote="chain_least-to-most_contextFull"
    # predictions_file = f'advanced_rag_results_{fileNote}.json'
    fileNote="k5_juristdiction-filter_gruende_withCritique_summary"
    predictions_file = f'naive_rag_case_predictions_{fileNote}.json'
    
    file_path = f'n_{fileNote}'
    actual_data = read_json_file(actual_file)
    predictions_data = read_json_file(predictions_file)

    if actual_data is None or predictions_data is None:
        print("Failed to load data.")
        return

    # Evaluate predictions and compute metrics
    metrics, y_true, y_pred = evaluate_predictions(actual_data, predictions_data)
    display_metrics(metrics)
    generate_charts(metrics, y_true, y_pred, file_path)

if __name__ == "__main__":
    main()

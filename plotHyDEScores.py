import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Read JSON file
def read_json(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

# Calculate average scores for each method
def calculate_averages(data):
    metrics = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"]
    method_scores = defaultdict(lambda: {metric: [] for metric in metrics})

    # Accumulate scores by method
    for case in data:
        method = case["method"]
        if "metrics" in case:
            for metric in metrics:
                method_scores[method][metric].append(case["metrics"].get(metric, 0))

    # Calculate averages
    averages = {}
    for method, scores in method_scores.items():
        averages[method] = {metric: sum(scores[metric]) / len(scores[metric]) if scores[metric] else 0 for metric in metrics}

    return averages

# Generate graphs for average scores by method
def plot_averages(averages, output_folder):
    metrics = averages["direct"].keys()
    methods = averages.keys()

    for metric in metrics:
        values = [averages[method][metric] for method in methods]

        plt.figure(figsize=(8, 5))
        plt.bar(methods, values, color=["#002147", "#4682b4", "#b0c4de"], alpha=0.7)
        plt.title(f"Average {metric} Scores by Method", fontsize=14)
        plt.xlabel("Method", fontsize=12)
        plt.ylabel(f"Average {metric} Score", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"Average_{metric}_Scores_by_Method.png"))
        plt.close()

# Generate summary statistics for each method
def generate_summary(averages, output_folder):
    summary_file = os.path.join(output_folder, "method_comparison_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as file:
        file.write("Average Scores by Method Across All Cases\n")
        file.write("=" * 40 + "\n")
        for method, metrics in averages.items():
            file.write(f"{method.capitalize()} Averages:\n")
            for metric, value in metrics.items():
                file.write(f"  {metric}: {value:.4f}\n")
            file.write("\n")

# Main function
def main():
    input_file = "hyde_comparison_results_batched_unbiased.json"  # Replace with your JSON filename
    output_folder = "output_method_comparison"
    os.makedirs(output_folder, exist_ok=True)

    # Read data
    data = read_json(input_file)

    # Calculate averages by method
    averages = calculate_averages(data)

    # Generate graphs
    plot_averages(averages, output_folder)

    # Generate summary
    generate_summary(averages, output_folder)

    print(f"Comparison completed. Results saved to {output_folder}/")

if __name__ == "__main__":
    main()
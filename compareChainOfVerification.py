import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Read JSON file
def read_json(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

# Extract and compare metrics
def extract_metrics(data):
    metrics = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"]
    
    transformed = {metric: [] for metric in metrics}
    generated = {metric: [] for metric in metrics}

    for case in data:
        if "metrics_transformed_actual" in case and "metrics_generated_actual" in case:
            for metric in metrics:
                transformed[metric].append(case["metrics_transformed_actual"].get(metric, 0))
                generated[metric].append(case["metrics_generated_actual"].get(metric, 0))

    # Compute averages
    avg_transformed = {metric: sum(values) / len(values) if values else 0 for metric, values in transformed.items()}
    avg_generated = {metric: sum(values) / len(values) if values else 0 for metric, values in generated.items()}

    return avg_transformed, avg_generated

# Plot comparison bar charts
def plot_comparison(avg_transformed, avg_generated, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    metrics = list(avg_transformed.keys())
    transformed_values = [avg_transformed[metric] for metric in metrics]
    generated_values = [avg_generated[metric] for metric in metrics]

    x = np.arange(len(metrics))  # Label locations
    width = 0.35  # Width of bars

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, transformed_values, width, label="Transformed Actual", color="#4682b4")
    plt.bar(x + width/2, generated_values, width, label="Generated Actual", color="#b0c4de")

    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Comparison of Transformed vs. Generated Metrics", fontsize=14)
    plt.xticks(x, metrics, rotation=25)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "metrics_comparison.png"))
    plt.close()

# Main function
def main():
    input_file = "chain-of-verification_evaluation_results_multi-query.json"  # Replace with your actual filename
    output_folder = "output_metrics_comparison_chain_of_eval"

    # Read data
    data = read_json(input_file)

    # Extract and compare averages
    avg_transformed, avg_generated = extract_metrics(data)

    # Generate comparison graph
    plot_comparison(avg_transformed, avg_generated, output_folder)

    print(f"Comparison completed. Results saved to {output_folder}/")

if __name__ == "__main__":
    main()

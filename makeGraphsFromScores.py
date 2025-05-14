import json
import matplotlib.pyplot as plt
import os
# Define colors
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
# Read JSON file
def read_json(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

# Calculate average scores by rank
def calculate_averages_by_rank(data):
    ranks = range(1, 11)  # Assuming ranks 1 to 10
    faiss_averages = {rank: {"BLEU": [], "ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": [], "METEOR": []} for rank in ranks}
    bm25_averages = {rank: {"BLEU": [], "ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": [], "METEOR": []} for rank in ranks}

    # Accumulate scores by rank
    for case in data:
        for retrieval in case["FAISS Retrievals"]:
            rank = retrieval["Rank"]
            for metric in faiss_averages[rank]:
                faiss_averages[rank][metric].append(retrieval[metric])

        for retrieval in case["BM25 Retrievals"]:
            rank = retrieval["Rank"]
            for metric in bm25_averages[rank]:
                bm25_averages[rank][metric].append(retrieval[metric])

    # Calculate averages for each rank
    for rank in ranks:
        for metric in faiss_averages[rank]:
            faiss_averages[rank][metric] = sum(faiss_averages[rank][metric]) / len(faiss_averages[rank][metric])
        for metric in bm25_averages[rank]:
            bm25_averages[rank][metric] = sum(bm25_averages[rank][metric]) / len(bm25_averages[rank][metric])

    return faiss_averages, bm25_averages

# Generate graphs for average scores by rank
def plot_averages_by_rank(faiss_averages, bm25_averages, output_folder):
    metrics = list(next(iter(faiss_averages.values())).keys())
    ranks = list(faiss_averages.keys())

    for metric in metrics:
        faiss_values = [faiss_averages[rank][metric] for rank in ranks]
        bm25_values = [bm25_averages[rank][metric] for rank in ranks]

        plt.figure(figsize=(10, 6))
        plt.plot(ranks, faiss_values, marker="o", label=f"FAISS - {metric}", color=COLORS["primary"]["midnight_blue"])
        plt.plot(ranks, bm25_values, marker="x", label=f"BM25 - {metric}", color=COLORS["primary"]["steel_blue"])
        plt.title(f"Average {metric} Scores by Rank (FAISS vs BM25)", fontsize=14)
        plt.xlabel("Rank", fontsize=12)
        plt.ylabel(f"{metric} Score", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"Average_{metric}_Scores_by_Rank.png"))
        plt.close()

# Generate summary statistics by rank
def generate_summary_by_rank(faiss_averages, bm25_averages, output_folder):
    summary_file = os.path.join(output_folder, "average_scores_by_rank_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as file:
        file.write("Average Scores by Rank Across All Cases\n")
        file.write("=" * 40 + "\n")
        file.write("FAISS Averages by Rank:\n")
        for rank, metrics in faiss_averages.items():
            file.write(f"  Rank {rank}:\n")
            for metric, value in metrics.items():
                file.write(f"    {metric}: {value:.4f}\n")
        file.write("\nBM25 Averages by Rank:\n")
        for rank, metrics in bm25_averages.items():
            file.write(f"  Rank {rank}:\n")
            for metric, value in metrics.items():
                file.write(f"    {metric}: {value:.4f}\n")

# Main function
def main():
    input_file = "bm25_vs_faiss_metrics_legacy.json"  # Replace with your JSON filename
    output_folder = "output_by_rank"
    os.makedirs(output_folder, exist_ok=True)

    # Read data
    data = read_json(input_file)

    # Calculate averages by rank
    faiss_averages, bm25_averages = calculate_averages_by_rank(data)

    # Generate graphs
    plot_averages_by_rank(faiss_averages, bm25_averages, output_folder)

    # Generate summary
    generate_summary_by_rank(faiss_averages, bm25_averages, output_folder)

if __name__ == "__main__":
    main()

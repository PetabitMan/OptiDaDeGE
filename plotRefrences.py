import json
import matplotlib.pyplot as plt
import re

# Define colors
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

def normalize_slug(slug):
    """Normalize slugs by removing letters and keeping only numbers."""
    return re.sub(r'[^0-9]', '', slug) if slug else ''

def load_data(file_path):
    """Load JSON data from the specified file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_comparison(data):
    """Analyze the comparison data by method and return statistics."""
    method_stats = {}

    for entry in data:
        method = entry.get("method", "Unknown")
        actual_refs = entry.get("actual_references", [])
        #hypothetical_refs = entry.get("hypothetical_references", [])
        hypothetical_refs = entry.get("transformed_references", [])

        # Normalize references assuming they are strings
        actual_set = {normalize_slug(ref) for ref in actual_refs}
        hypothetical_set = {normalize_slug(ref) for ref in hypothetical_refs}

        matched = actual_set & hypothetical_set
        only_in_actual = actual_set - hypothetical_set
        only_in_hypothetical = hypothetical_set - actual_set

        if method not in method_stats:
            method_stats[method] = {
                "total_matched": 0,
                "total_only_in_actual": 0,
                "total_only_in_hypothetical": 0,
            }

        method_stats[method]["total_matched"] += len(matched)
        method_stats[method]["total_only_in_actual"] += len(only_in_actual)
        method_stats[method]["total_only_in_hypothetical"] += len(only_in_hypothetical)

    return method_stats


def plot_statistics_by_method(stats):
    """Generate bar charts for the comparison statistics by method."""
    methods = list(stats.keys())
    matched = [stats[method]["total_matched"] for method in methods]
    only_in_actual = [stats[method]["total_only_in_actual"] for method in methods]
    only_in_hypothetical = [stats[method]["total_only_in_hypothetical"] for method in methods]

    x = range(len(methods))

    # Create the plot
    plt.figure(figsize=(12, 7))
    plt.bar(x, matched, width=0.25, label="Matched", color=COLORS["primary"]["steel_blue"])
    plt.bar([p + 0.25 for p in x], only_in_actual, width=0.25, label="Only in Actual", color=COLORS["secondary"]["light_blue"])
    plt.bar([p + 0.5 for p in x], only_in_hypothetical, width=0.25, label="Only in Hypothetical", color=COLORS["primary"]["midnight_blue"])

    # Style the plot
    plt.title("Comparison of Legal References by Method", fontsize=16)
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks([p + 0.25 for p in x], methods, color=COLORS["accent"]["dark_gray"])
    plt.yticks(color=COLORS["accent"]["dark_gray"])
    plt.grid(axis="y", color=COLORS["secondary"]["light_gray"], linestyle="--", linewidth=0.7)
    plt.legend()

    # Show and save the plot
    plt.tight_layout()
    plt.savefig("comparison_statistics_by_method_chain.png")
    plt.show()

def main():
    """Main function to load, analyze, and visualize data."""
    input_file = "reference_comparison_chain_results.json"  # Update with your file path
    try:
        data = load_data(input_file)
        stats = analyze_comparison(data)
        print("Statistics by Method:", stats)
        plot_statistics_by_method(stats)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


import json
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define your color palette
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

# Specify the path to your cases file
cases_file_path = 'cases_with_references_test.json'  # Replace with your actual file path

# Specify the output path for the plot
output_plot_path = 'law_references_frequency.png'

# Load the cases file
print("Loading cases file...")
with open(cases_file_path, 'r', encoding='utf-8') as f:
    cases = json.load(f)

# Initialize variables
case_references = []
cases_with_references = 0

# Extract case references with a progress bar
print("Extracting case references...")
for case in tqdm(cases, desc="Processing cases"):
    references = case.get('simple_refs', [])
    if references:
        cases_with_references += 1
    case_references.extend(references)

# Count occurrences
print("Counting references...")
reference_counts = Counter(case_references)

# Metrics
total_references = len(case_references)
unique_references = len(reference_counts)
total_cases = len(cases)

print("\nMetrics:")
print(f"Total cases processed: {total_cases}")
print(f"Cases with references: {cases_with_references}")
print(f"Total references: {total_references}")
print(f"Unique references: {unique_references}")

# Sort by frequency
sorted_counts = reference_counts.most_common()

# Separate labels and counts for plotting
_, counts = zip(*sorted_counts)

# Select top N references for zoomed-in view
top_n = 20
top_labels, top_counts = zip(*sorted_counts[:top_n])

# Create combined plot
print(f"Saving plot to {output_plot_path}...")
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Plot 1: Full distribution
axs[0].bar(range(len(counts)), counts, color=COLORS['primary']['steel_blue'])
axs[0].set_title('Frequency of Case References (Full Distribution)', color=COLORS['primary']['midnight_blue'])
axs[0].set_xlabel('Index (Each represents a unique case reference)', color=COLORS['primary']['midnight_blue'])
axs[0].set_ylabel('Frequency', color=COLORS['primary']['midnight_blue'])
axs[0].tick_params(axis='x', colors=COLORS['accent']['dark_gray'])
axs[0].tick_params(axis='y', colors=COLORS['accent']['dark_gray'])
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['left'].set_color(COLORS['secondary']['light_gray'])
axs[0].spines['bottom'].set_color(COLORS['secondary']['light_gray'])

# Plot 2: Top N references
axs[1].barh(top_labels, top_counts, color=COLORS['secondary']['light_blue'])
axs[1].invert_yaxis()
axs[1].set_title(f'Top {top_n} Case References', color=COLORS['primary']['midnight_blue'])
axs[1].set_xlabel('Frequency', color=COLORS['primary']['midnight_blue'])
axs[1].set_ylabel('Case Reference', color=COLORS['primary']['midnight_blue'])
axs[1].tick_params(axis='x', colors=COLORS['accent']['dark_gray'])
axs[1].tick_params(axis='y', colors=COLORS['accent']['dark_gray'])
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['left'].set_color(COLORS['secondary']['light_gray'])
axs[1].spines['bottom'].set_color(COLORS['secondary']['light_gray'])

# Adjust layout
plt.tight_layout()

# Set background color for the figure
fig.patch.set_facecolor(COLORS['accent']['white'])

# Save the plot
plt.savefig(output_plot_path, facecolor=fig.get_facecolor())
plt.close()

print(f"\nPlot saved successfully to {output_plot_path}")

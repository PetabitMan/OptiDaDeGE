import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm

# Specify the path to your cases file
cases_file_path = 'sufficient_cases.json'  # Replace with your actual file path

# Load the cases file
print("Loading cases file...")
with open(cases_file_path, 'r', encoding='utf-8') as f:
    cases = json.load(f)

# Extract case references with a progress bar
print("Extracting case references...")
case_references = []
for case in tqdm(cases, desc="Processing cases"):
    references = case.get('case_references', [])
    case_references.extend(references)

# Count occurrences and rank frequencies
print("Counting references...")
reference_counts = Counter(case_references)
sorted_counts = sorted(reference_counts.values(), reverse=True)
ranks = range(1, len(sorted_counts) + 1)

# Convert to log-log scale
log_ranks = np.log(ranks)
log_frequencies = np.log(sorted_counts)

# Fit a linear model to log-log data
slope, intercept, r_value, p_value, std_err = linregress(log_ranks, log_frequencies)

# Plot the data and the fitted line
# Plot the data and the fitted line in zwei nebeneinander stehende Subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Linker Plot: nur die Datenpunkte
axes[0].scatter(log_ranks, log_frequencies, alpha=0.6)
axes[0].set_title('Daten (log-log)')
axes[0].set_xlabel('log(Rang)')
axes[0].set_ylabel('log(Häufigkeit)')

# Rechter Plot: Datenpunkte plus Fit-Linie
axes[1].scatter(log_ranks, log_frequencies, alpha=0.6, label='Daten (log-log)')
axes[1].plot(log_ranks, slope * log_ranks + intercept,
             linestyle='-', linewidth=2, label=f'Fit (Steigung={slope:.2f})')
axes[1].set_title('Fit der Linearen Regression')
axes[1].set_xlabel('log(Rang)')
axes[1].set_ylabel('log(Häufigkeit)')
axes[1].legend()

plt.tight_layout()

# Save the combined figure
output_plot_path = 'zipf_analysis_side_by_side.png'
plt.savefig(output_plot_path)
plt.close()

print(f"\nAnalysis completed. Plot saved to {output_plot_path}")
print("Zipf's Law Fit:")
print(f"  Slope (Scaling Exponent): {slope:.2f}")
print(f"  R-squared: {r_value**2:.2f}")
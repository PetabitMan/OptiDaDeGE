import json
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

# Pfad zu Deiner JSON-Datei
cases_file_path = 'sufficient_cases.json'

# Lade die Daten
with open(cases_file_path, 'r', encoding='utf-8') as f:
    cases = json.load(f)

# Sammle alle Referenzen
case_references = []
for case in tqdm(cases, desc="Processing cases"):
    case_references.extend(case.get('case_references', []))

# Zähle Häufigkeiten
reference_counts = Counter(case_references)
sorted_counts = reference_counts.most_common()
_, counts = zip(*sorted_counts)

# Top-20 für Zoom
top_n = 20
top_labels, top_counts = zip(*sorted_counts[:top_n])

# ------ Plot 1: Full Distribution ------
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(range(len(counts)), counts, color='#4682B4')
ax1.set_title('Frequency of Case References (Full Distribution)', color='black')
ax1.set_xlabel('Index (Each represents a unique case reference)', color='black')
ax1.set_ylabel('Frequency', color='black')
ax1.tick_params(axis='x', colors='black')
ax1.tick_params(axis='y', colors='black')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#D3D3D3')
ax1.spines['bottom'].set_color('#D3D3D3')
fig1.patch.set_facecolor('white')
fig1.tight_layout()
fig1.savefig('full_distribution_black_text.png', facecolor=fig1.get_facecolor())
plt.close(fig1)

# ------ Plot 2: Top-N References ------
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.barh(top_labels, top_counts, color='#B0C4DE')
ax2.invert_yaxis()
ax2.set_title(f'Top {top_n} Case References', color='black')
ax2.set_xlabel('Frequency', color='black')
ax2.set_ylabel('Case Reference', color='black')
ax2.tick_params(axis='x', colors='black')
ax2.tick_params(axis='y', colors='black')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#D3D3D3')
ax2.spines['bottom'].set_color('#D3D3D3')
fig2.patch.set_facecolor('white')
fig2.tight_layout()
fig2.savefig('top20_references_black_text.png', facecolor=fig2.get_facecolor())
plt.close(fig2)

print("\nPlots saved as 'full_distribution_black_text.png' and 'top20_references_black_text.png'")

import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# Spezifiziere den Pfad zur Datei mit den Fällen
cases_file_path = 'enriched_cases.json'  # Ersetze dies durch deinen Dateipfad

# Lade die Fälle
print("Lade die Fall-Datei...")
with open(cases_file_path, 'r', encoding='utf-8') as f:
    cases = json.load(f)

# Daten vorbereiten
winners = {"Kläger": [], "Angeklagter": []}

print("Verarbeite die Fälle...")
for case in tqdm(cases, desc="Fälle verarbeiten"):
    winner = case.get("winner")
    costs = case.get("costs_borne_by", {})
    prozent_kläger = costs.get("prozentzahl_kläger", None)
    prozent_angeklagte = costs.get("prozentzahl_angeklagte", None)
    
    if winner == "Kläger":
        winners["Kläger"].append((prozent_kläger, prozent_angeklagte))
    elif winner == "Angeklagter":
        winners["Angeklagter"].append((prozent_kläger, prozent_angeklagte))

# Daten für den Plot vorbereiten
labels = []
prozent_kläger_values = []
prozent_angeklagte_values = []

for winner, values in winners.items():
    labels.extend([winner] * len(values))
    for kläger, angeklagter in values:
        prozent_kläger_values.append(kläger)
        prozent_angeklagte_values.append(angeklagter)

# Plot erstellen
plt.figure(figsize=(10, 6))
plt.scatter(prozent_kläger_values, prozent_angeklagte_values, alpha=0.6, c=["blue" if l == "Kläger" else "red" for l in labels], label="Kläger/Angeklagter")
plt.title("Verteilung der Kostenaufteilung nach Gewinner")
plt.xlabel("Prozentzahl Kosten (Kläger)")
plt.ylabel("Prozentzahl Kosten (Angeklagter)")
plt.axhline(50, color="gray", linestyle="--", alpha=0.5)
plt.axvline(50, color="gray", linestyle="--", alpha=0.5)
plt.legend(["Kläger", "Angeklagter"], loc="upper right")
plt.tight_layout()

# Speichere den Plot
output_plot_path = 'cases_winner_costs_plot.png'
plt.savefig(output_plot_path)
plt.close()

print(f"Plot wurde gespeichert unter {output_plot_path}")

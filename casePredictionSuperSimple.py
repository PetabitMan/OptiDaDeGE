import json
import os
from pydantic import BaseModel
from typing import Union
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()
correct_winner_count = 0  # Global counter for correct winner predictions
total_cases = 0  # Counter for total cases processed

# Pydantic-Modell für das strukturierte JSON-Format
class CostBorneBy(BaseModel):
    prozentzahl_kläger: Union[float, None]
    prozentzahl_angeklagte: Union[float, None]

class CourtCaseAnalysis(BaseModel):
    winner: str
    confidence_notes: str
#Experimente
#Komplexität reduzieren
#System Prompt auf neutralität polen
#CoT
def call_openai_api_with_structure(tatbestand, enrichment):
    try:
        enrichment_block = (
            f"\n\nZusätzliche Informationen zur Entscheidungsfindung:\n{enrichment}"
            if enrichment
            else ""
        )
        # API-Aufruf mit Structured Outputs
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            temperature=0,       # Für deterministische und strukturierte Ausgabe
            max_tokens=300,      # Ausreichend Tokens, um den kompletten JSON-Output zu generieren
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,  # Modell mit Structured Outputs Support
            messages=[
                {
                    "role": "system",
                    "content": """
                    Du bist ein hochspezialisierter juristischer Assistent. Deine Aufgabe ist es, auf Basis des angegebenen Tatbestands eine strukturierte Vorhersage des Tenors im JSON-Format zu erstellen. 
                    Arbeite stets präzise, konsistent und im Rahmen der juristischen Logik. Falls Unsicherheiten bestehen, dokumentiere diese klar.                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    Analysiere den folgenden Tatbestand und gib den wahrscheinlichsten Tenor im folgenden JSON-Format zurück:
                    ```json
                    {{
                        "winner": "Beklagter/Kläger",
                        "confidence_notes": "Erklärungen zu möglichen Unsicherheiten"
                    }}
                    ```
                    
                    Hier ist der Tatbestand:
                    {tatbestand}
                    {enrichment_block}
                    """
                }
            ],
            response_format=CourtCaseAnalysis,  # Pydantic-Modell für strukturierte Ausgabe
        )

        return response.choices[0].message.parsed

    except Exception as e:
        print(f"Error during API call: {e}")
        return None


# JSON-Datei einlesen
def load_cases(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: JSON decoding failed: {e}")
        return []

# JSON-Datei speichern
def save_cases(file_path, cases):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(cases, file, ensure_ascii=False, indent=4)
        print("Die JSON-Datei wurde erfolgreich aktualisiert.")
    except Exception as e:
        print(f"Error while saving JSON file: {e}")

# Hauptprozess zur Vorhersage und zum Vergleich
def process_case_predictions(input_file, output_file):
    cases = load_cases(input_file)
    if not cases:
        print("Keine Fälle gefunden.")
        return
    #Load in Data as desired in here
    enrichment=""
    correct_winner_count = 0
    total_cases = 0

    for case in tqdm(cases[:100], desc="Processing Cases"):
        tatbestand = case.get("structured_content", {}).get("tatbestand")
        if tatbestand:
            #print(f"Vorhersage für Fall: {tatbestand[:50]}...")  # Kürze den Tatbestand zur Übersicht
            predicted_tenor = call_openai_api_with_structure(tatbestand, enrichment)
            actual_data = {
                "winner": case.get("winner"),
            }

            if predicted_tenor:
                case["predicted_tenor"] = predicted_tenor.dict()  # Pydantic-Objekt in Dict konvertieren
                comparison_result = compare_tenors(predicted_tenor, actual_data)
                case["comparison_result"] = comparison_result

                # Count correct predictions for winner
                if predicted_tenor.winner == actual_data["winner"]:
                    correct_winner_count += 1

                total_cases += 1

    save_cases(output_file, cases)

    # Print summary statistics
    print(f"Total cases processed: {total_cases}")
    print(f"Correct winner predictions: {correct_winner_count}")
    if total_cases > 0:
        print(f"Accuracy of winner prediction: {correct_winner_count / total_cases * 100:.2f}%")




# Funktion zum Vergleich von vorhergesagtem und tatsächlichem Fall
def compare_tenors(predicted, actual):
    if not actual:
        return "No actual data available for comparison."
    
    differences = []
    # Compare the winner
    if predicted.winner != actual.get("winner"):
        differences.append(f"Winner mismatch: predicted {predicted.winner}, actual {actual.get('winner')}")
    
    

    return {
        "status": "Matches all fields" if not differences else "Mismatches found",
        "details": differences,
    }






# Eingabe- und Ausgabedateien
input_file = 'enriched_cases.json'
output_file = 'baseline_case_predictions_simple_parameter_tuning.json'

# Prozess starten
process_case_predictions(input_file, output_file)






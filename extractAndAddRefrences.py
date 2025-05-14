import re
import json
from tqdm import tqdm
# Funktion zum Laden von Buch-Codes aus einer JSON-Datei
def load_book_codes(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Fehler beim Laden der Buch-Codes: {e}")
        return []

# Funktion zur Extraktion von Rechtsreferenzen
def extract_law_references(text, book_codes, max_length=50):
    # Kombiniere die Buch-Codes in ein Regex-Muster
    book_codes_pattern = r"|".join(map(re.escape, book_codes))
    # Regex-Muster für Rechtsreferenzen
    statute_pattern = rf"(§{{1,2}}.*?\b(?:{book_codes_pattern})\b)"
    # Suche nach allen Übereinstimmungen
    matches = re.findall(statute_pattern, text)
    # Bereinige und filtere die Übereinstimmungen
    cleaned_matches = [
        " ".join(match.split()) for match in matches if len(match) <= max_length
    ]
    return cleaned_matches
# Funktion zur Vereinfachung der Rechtsreferenzen unter Berücksichtigung der Komma-Regel
def simplify_references(references):
    simplified = set()  # Verwende ein Set, um Duplikate zu vermeiden
    for ref in references:
        # Regex zum Extrahieren von Nummern und Buch-Codes
        match = re.search(r"§§?\s*([\d\s,]+).*?(\b\S+$)", ref)
        
        numbers = []
        if match:
            parts = ref.split(",")
            for part in parts: 
                numbers_part = re.search(r"\b\d+(?=\D|\b)", part)
                if numbers_part:
                    numbers.append(numbers_part.group()) # Fängt alle Zahlen und Kommas ein
            book_code = match.group(2)    # Der Buch-Code (z. B. ZPO)
            
    
            for num in numbers:
                if num.isdigit():  # Überprüfe, ob es eine gültige Zahl ist
                    simplified.add(f"{num} {book_code}")

    return list(simplified)

# Hauptprozess zur Bearbeitung der Fälle
def process_cases(input_file, output_file, book_codes_file):
    # Lade die Buch-Codes
    print("Loading Book Codes")

    book_codes = load_book_codes(book_codes_file)
    if not book_codes:
        print("Keine Buch-Codes gefunden. Beende.")
        return

    try:
        # Lade die Fälle aus der Eingabedatei
        with open(input_file, 'r', encoding='utf-8') as file:
            cases = json.load(file)

        for case in tqdm(cases, desc="Processing Cases"):
            # Extrahiere den Text der Entscheidungsgründe
            entscheidungsgruende = case.get("structured_content", {}).get("entscheidungsgruende", "")
            if entscheidungsgruende:
                # Finde Rechtsreferenzen in den Entscheidungsgründen
                references = extract_law_references(entscheidungsgruende, book_codes)
                simplified_refs = simplify_references(references)
                # Füge die Referenzen in den Fall ein
                case["law_references"] = references
                case["simple_refs"] = simplified_refs

        # Speichere die aktualisierten Fälle in die Ausgabedatei
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(cases, file, ensure_ascii=False, indent=4)
        print(f"Verarbeitete Fälle gespeichert in {output_file}")

    except Exception as e:
        print(f"Fehler während der Verarbeitung: {e}")

# Beispiel-Aufruf
if __name__ == "__main__":
    input_file = "sufficient_cases.json"  # Pfad zur Eingabedatei
    output_file = "sufficient_cases.json"  # Pfad zur Ausgabedatei
    book_codes_file = "book_codes.json"  # Pfad zur Buch-Codes-Datei
    process_cases(input_file, output_file, book_codes_file)


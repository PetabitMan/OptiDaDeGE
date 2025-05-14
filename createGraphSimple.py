import os
import json
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI



import openai
# --- Environment Setup ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI()

# --- Simple Models ---

class TatbestandsmerkmalModel(BaseModel):
    name: str
    question: str

class TatbestandsmerkmaleResponse(BaseModel):
    tatbestandsmerkmale: List[TatbestandsmerkmalModel]

class RechtsfolgeModel(BaseModel):
    art: str

# --- Functions ---

def load_laws(file_path: str) -> List[dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

def save_laws(data: List[dict], file_path: str):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Saved processed laws to {file_path}")
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")

def call_openai_extract_tbm(law_content: str) -> Optional[List[TatbestandsmerkmalModel]]:
    try:
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
Du bist ein juristischer Assistent.

Deine Aufgabe ist es, einen deutschen Gesetzestext zu analysieren und folgende Informationen in einem strukturierten JSON-Format aufzubereiten:

 **Tatbestandsmerkmale**
   - Gib für jedes Tatbestandsmerkmal eine kurze, präzise Überschrift an ("name").
   - Formuliere für jedes Tatbestandsmerkmal eine konkrete, fallbezogene Ja/Nein-Frage ("question"), die auf einen bestimmten Sachverhalt oder eine Handlung der betroffenen Person abzielt.
   - Die Fragen dürfen sich nicht auf die Existenz einer gesetzlichen Regelung beziehen, sondern müssen die tatsächlichen Umstände im Einzelfall abprüfen.
   - Vermeide Modalverben ("sollen", "dürfen", "können"). Nutze klare Tatsachenfragen wie "Hat X getan?", "Besteht Y?", "Liegt Z vor?".


**Format:**
```json
{
  "tatbestandsmerkmale": [
    {
      "name": "Kurzer Titel des Merkmals",
      "question": "Konkrete Ja/Nein-Frage"
    }, 
  ]
}

}
```
"""
                },
                {"role": "user", "content": f"Hier ist der Gesetzestext:\n{law_content}"}
            ],
            response_format=TatbestandsmerkmaleResponse,
        )
        return response.choices[0].message.parsed if response else None
    except Exception as e:
        print(f"Error in extract_tbm: {e}")
        return None

def call_openai_extract_rechtsfolge(law_content: str) -> Optional[RechtsfolgeModel]:
    try:
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
                Du bist ein juristischer Assistent.

                Deine Aufgabe ist es, einen deutschen Gesetzestext zu analysieren und folgende Informationen in einem strukturierten JSON-Format aufzubereiten:

                **Rechtsfolge**
                - Gib die Rechtsfolge kurz, klar und aktiv formuliert an ("art").
                - Beschreibe die Rechtsfolge so, dass sie sich auf die betroffene Person oder den konkreten Rechtsstatus bezieht.
                - Vermeide komplizierte oder lange Formulierungen.

                Antwortformat:
                ```json
                {
                "art": "Kurze, aktive Beschreibung der Rechtsfolge"
                }
                ```
"""
                },
                {"role": "user", "content": f"Hier ist der Gesetzestext:\n{law_content}"}
            ],
            response_format=RechtsfolgeModel,
        )
        return response.choices[0].message.parsed if response else None
    except Exception as e:
        print(f"Error in extract_rechtsfolge: {e}")
        return None

def process_law(law: dict) -> dict:
    law_content = law.get("content", "")
    if not law_content:
        print(f"Law ID {law.get('id', 'unknown')} has no content.")
        return law

    tbm_extracted = call_openai_extract_tbm(law_content)
    rechtsfolge_extracted = call_openai_extract_rechtsfolge(law_content)

    new_law = {
        "id": law.get("id"),
        "book": law.get("book"),
        "title": law.get("title"),
        "slug": law.get("slug"),
        "section": law.get("section"),
        "tatbestandsmerkmale": [tbm.dict() for tbm in tbm_extracted.tatbestandsmerkmale] if tbm_extracted else [],
        "rechtsfolge": rechtsfolge_extracted.dict() if rechtsfolge_extracted else {},
        "original_text": law_content  # <-- hier neu dazu!
    }
    return new_law

def pruefe_norm(law: dict):
    print(f"\nPrüfung von {law['title']} ({law['section']})")
    all_passed = True
    for tbm in law.get("tatbestandsmerkmale", []):
        answer = input(f"{tbm['question']} (ja/nein): ").strip().lower()
        if answer != "ja":
            all_passed = False
    if all_passed:
        print(f"Alle Tatbestandsmerkmale erfüllt. Rechtsfolge: {law['rechtsfolge'].get('art', 'Unbekannt')}")
    else:
        print("Mindestens ein Tatbestandsmerkmal ist nicht erfüllt. Rechtsfolge tritt nicht ein.")

# --- Main Execution ---

if __name__ == "__main__":
    input_file = "laws.jsonl"
    output_file = "processed_laws.json"

    laws = load_laws(input_file)
    if not laws:
        print("No laws loaded.")
        exit(1)

    processed_laws = []
    count = 0
    for law in tqdm(laws, desc="Processing Laws"):
        if law.get("book") != 1280:
            continue
        if law.get("title") == "(weggefallen)":
            continue
        if law.get("slug") in {"eingangsformel", "inhalt", "inhaltsubersicht", "inhaltsverzeichnis"}:
            continue
        if count > 2:
            break

        processed_law = process_law(law)
        count += 1
        processed_laws.append(processed_law)

    save_laws(processed_laws, output_file)

    # Example: Prüfung eines Gesetzes starten
    if processed_laws:
        pruefe_norm(processed_laws[0])

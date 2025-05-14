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

# --- Pydantic-Modelle für strukturierte Ausgaben ---

class TatbestandsmerkmalModel(BaseModel):
    id: str
    name: str
    criteria: str
    typ: str  # z.B. "objektiv" oder "subjektiv"
    question: str # <--- neu

class TatbestandsmerkmaleResponse(BaseModel):
    tatbestandsmerkmale: List[TatbestandsmerkmalModel]

class RechtsfolgeModel(BaseModel):
    id: str
    art: str
    conditions: str

# --- Knowledge Graph Klassen ---

class Norm:
    def __init__(self, book_code: str, slug: str, title: str, additional_info: dict = None):
        self.book_code = book_code
        self.slug = slug
        self.title = title
        self.tatbestandsmerkmale: List[Tatbestandsmerkmal] = []
        self.rechtsfolge: Optional[Rechtsfolge] = None
        self.additional_info = additional_info or {}

    def __repr__(self):
        return f"Norm({self.book_code}, {self.slug}, {self.title})"

class Tatbestandsmerkmal:
    def __init__(self, tbm: TatbestandsmerkmalModel):
        self.tbm_id = tbm.id
        self.name = tbm.name
        self.criteria = tbm.criteria
        self.typ = tbm.typ

    def __repr__(self):
        return f"TBM({self.tbm_id}, {self.name})"

class Rechtsfolge:
    def __init__(self, rf: RechtsfolgeModel):
        self.rf_id = rf.id
        self.art = rf.art
        self.conditions = rf.conditions

    def __repr__(self):
        return f"Rechtsfolge({self.rf_id}, {self.art})"

class KnowledgeGraph:
    def __init__(self):
        self.norms = {}  # key: (book_code, slug), value: Norm-Objekt

    def add_norm(self, norm: Norm):
        key = f"{norm.book_code}_{norm.slug}"
        self.norms[key] = norm

    def __repr__(self):
        return f"KnowledgeGraph(Norms: {len(self.norms)})"

# --- Funktionen zum Laden/Speichern von Laws bzw. Graph ---

def load_laws(file_path: str) -> List[dict]:
    """
    Liest eine .jsonl-Datei ein und gibt eine Liste von JSON-Objekten zurück.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

def save_graph(graph: KnowledgeGraph, file_path: str):
    """
    Speichert den Knowledge Graph als JSON in der angegebenen Datei.
    """
    try:
        data = {
            "norms": {
                key: {
                    "title": norm.title,
                    "tatbestandsmerkmale": [
                        {"id": tbm.tbm_id, "name": tbm.name, "criteria": tbm.criteria, "typ": tbm.typ}
                        for tbm in norm.tatbestandsmerkmale
                    ] if norm.tatbestandsmerkmale else [],
                    "rechtsfolge": {
                        "id": norm.rechtsfolge.rf_id,
                        "art": norm.rechtsfolge.art,
                        "conditions": norm.rechtsfolge.conditions
                    } if norm.rechtsfolge else None
                }
                for key, norm in graph.norms.items()
            }
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("Graph saved successfully.")
    except Exception as e:
        print(f"Error saving graph: {e}")

# --- Funktionen für API-Aufrufe (jeweils getrennt) ---

def call_openai_api_extract_tbm(law_content: str) -> Optional[List[TatbestandsmerkmalModel]]:
    """
    Extrahiert aus dem Gesetzestext die relevanten Tatbestandsmerkmale als Liste von JSON-Objekten.
    """
    try:
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
                        Du bist ein juristischer Assistent. Analysiere den folgenden Gesetzestext und extrahiere alle relevanten Tatbestandsmerkmale.

                        Für jedes Tatbestandsmerkmal:
                        - Gib eine klare, konkrete Ja/Nein-Frage an, mit der geprüft werden kann, ob das Merkmal erfüllt ist.
                        - Verwende einfache und präzise Formulierungen, wie ein Jurist sie nutzen würde.

                        Antwortformat:
                        ```json
                        {
                        "tatbestandsmerkmale": [
                            {
                            "id": "TBM1",
                            "name": "Bezeichnung des Merkmals",
                            "criteria": "Beschreibung der Kriterien",
                            "typ": "objektiv",
                            "question": "Klare Ja/Nein Frage zur Prüfung des Merkmals"
                            }
                        ]
                        }
                    }
                    ```
                    Falls keine Tatbestandsmerkmale gefunden werden, gib ein leeres Array zurück.
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

def call_openai_api_extract_rechtsfolge(law_content: str) -> Optional[RechtsfolgeModel]:
    """
    Extrahiert aus dem Gesetzestext die Rechtsfolge als strukturiertes JSON.
    """
    try:
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
                    Du bist ein hochspezialisierter juristischer Assistent. Bitte analysiere den folgenden Gesetzestext und extrahiere die Rechtsfolge.
                    Gib die Antwort im folgenden JSON-Format zurück:
                    ```json
                    {
                      "id": "RF1",
                      "art": "Bezeichnung der Rechtsfolge",
                      "conditions": "Bedingungen, unter denen die Rechtsfolge eintritt"
                    }
                    ```
                    Falls keine Rechtsfolge angegeben ist, gib leere Strings zurück.
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


def process_law_entry(graph: KnowledgeGraph, law_json: dict):
    law_content = law_json.get("content", "")
    if not law_content:
        print("Kein 'content' im Law-JSON gefunden.")
        return

    book_code = law_json.get("book", "unknown")
    slug = law_json.get("slug", "unknown")
    title = law_json.get("title", "")

    norm_obj = Norm(book_code, slug, title)

    tbm_extracted = call_openai_api_extract_tbm(law_content)
    rechtsfolge_extracted = call_openai_api_extract_rechtsfolge(law_content)
    # Falls die Antwort eine Instanz von TatbestandsmerkmaleResponse ist, extrahiere die Liste
    if isinstance(tbm_extracted, TatbestandsmerkmaleResponse):
        tbm_extracted = tbm_extracted.tatbestandsmerkmale
    print(f"DEBUG: Tatbestandsmerkmale extrahiert ({len(tbm_extracted)} Merkmale): {tbm_extracted}")


    for tbm in tbm_extracted:
        if not isinstance(tbm, TatbestandsmerkmalModel):
            print(f"Fehler: Element ist kein TatbestandsmerkmalModel! {tbm}")
            continue  # Überspringe fehlerhafte Elemente

        norm_obj.tatbestandsmerkmale.append(Tatbestandsmerkmal(tbm))

    print(f"DEBUG: Tatbestandsmerkmale nach Umwandlung: {norm_obj.tatbestandsmerkmale}")
    if rechtsfolge_extracted:
        norm_obj.rechtsfolge = Rechtsfolge(rechtsfolge_extracted)

    graph.add_norm(norm_obj)
    print(f"Law mit ID {law_json.get('id', 'unbekannt')} wurde in den Graph eingefügt.")

# --- Main Entry Point ---

if __name__ == "__main__":
    laws = load_laws("laws.jsonl")
    if not laws:
        print("Keine Laws gefunden. Bitte prüfe die Datei.")
        exit(1)

    kg = KnowledgeGraph()
    count = 0
    for law in tqdm(laws, desc="Processing Laws"):
        if law.get("book", "unknown") != 1280:
            continue
        if law.get("title", "unknown") == "(weggefallen)":
            continue
        if law.get("slug", "unknown") == "eingangsformel":
            continue
        if law.get("slug", "unknown") == "inhalt":
            continue
        if law.get("slug", "unknown") == "inhaltsubersicht":
            continue
        if law.get("slug", "unknown") == "inhaltsverzeichnis":
            continue
        if count > 3:
            break
        process_law_entry(kg, law)
        count += 1

    save_graph(kg, "knowledge_graph_BGB.json")
    print("Finaler Graph-Zustand:")
    print(kg)

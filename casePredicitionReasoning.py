import os
import sys
import json
from typing import Union
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import tiktoken

# === Setup ===
load_dotenv()
client = OpenAI()

class CourtCaseAnalysis(BaseModel):
    winner: str
    confidence_notes: str

# === Token- und Kostenkalkulation ===

def count_tokens(text: str, model: str = "o4-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def estimate_cost(
    system_prompt: str,
    cases: list,
    n_cases: int = 100,
    max_output_tokens: int = 2000,
    model: str = "o4-mini"
) -> float:
    tokens_sys = count_tokens(system_prompt, model)
    total = 0.0
    # o4-mini Pricing: 1.10 $/1M input, 4.40 $/1M output (inkl. reasoning_tokens)
    for case in cases[:n_cases]:
        tb = case.get("structured_content", {}).get("tatbestand", "")
        tokens_in = tokens_sys + count_tokens(tb, model)
        total += 1.10 * (tokens_in / 1e6) + 4.40 * (max_output_tokens / 1e6)
    return total

# === API-Call mit Responses API ===

def call_openai_api_with_structure(tatbestand: str, enrichment: str = None):
    """
    o4-mini über Responses API aufrufen mit:
      - auto-retry bei incomplete
      - Logging des Abbruchgrunds
      - Salvage-Logik für unvollständiges JSON
    """
    system_msg = """
Du bist ein hochspezialisierter juristischer Assistent. Deine Aufgabe ist es, auf Basis
des angegebenen Tatbestands eine strukturierte Vorhersage des Tenors im JSON-Format zu erstellen.
Arbeite stets präzise, konsistent und im Rahmen der juristischen Logik. Falls Unsicherheiten
bestehen, dokumentiere diese klar.
Gib die strukturierte Vorhersage im JSON-Format
   {{
                        "winner": "Beklagter/Kläger",
                        "confidence_notes": "Erklärungen zu möglichen Unsicherheiten"
                    }}.
    """
    user_msg = tatbestand
    if enrichment:
        user_msg += "\n\nZusätzliche Informationen:\n" + enrichment

    def _call(effort_level: str, max_out: int):
        return client.responses.create(
            model="o4-mini",
            reasoning={"effort": effort_level},
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg}
            ],
            max_output_tokens=max_out,
        )

    # 1) Erster Versuch
    try:
        resp = _call("low", 2000)
    except Exception as e:
        print("API-Fehler beim Aufruf:", e)
        return None

    # 2) Auto-Retry bei incomplete wegen max_output_tokens
    if getattr(resp, "status", None) == "incomplete":
        reason = getattr(resp.incomplete_details, "reason", "unknown")
        print(f"Incomplete (Grund: {reason}).")
        if reason == "max_output_tokens":
            print("→ Erhöhe auf 5000 und retry …")
            try:
                resp = _call("low", 5000)
            except Exception as e:
                print("API-Fehler beim Retry:", e)
                return None

    # 3) Zweiter Retry mit niedrigem effort, falls noch immer incomplete
    if getattr(resp, "status", None) == "incomplete":
        reason = getattr(resp.incomplete_details, "reason", "unknown")
        print(f"Noch immer incomplete (Grund: {reason}). Versuche effort='low' …")
        try:
            resp = _call("low", 5000)
        except Exception as e:
            print("API-Fehler beim zweiten Retry:", e)
            return None

    # 4) Leere Ausgabe abfangen
    if not resp.output_text:
        print("Keine sichtbare Ausgabe, breche ab.")
        return None

    raw = resp.output_text

    # 5) Versuch, direkt zu parsen
    try:
        return CourtCaseAnalysis.parse_raw(raw)
    except Exception as e:
        print("Fehler beim Parsen der JSON-Antwort:")
        print(">>>", raw)
        print(e)

    # 6) Salvage: JSON zwischen erstem '{' und letztem '}' extrahieren
    start = raw.find('{')
    end   = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end+1]
        print("Versuche Salvage-Parse auf:", candidate)
        try:
            return CourtCaseAnalysis.parse_raw(candidate)
        except Exception as e2:
            print("Salvage parsing fehlgeschlagen:", e2)

    # 7) Wenn alles scheitert, None zurückgeben
    print("Konnte kein valides JSON extrahieren.")
    return None


# === Vergleich & I/O ===

def compare_tenors(predicted, actual_winner: str):
    if not predicted:
        return {"status": "No prediction", "details": []}
    diffs = []
    if predicted.winner != actual_winner:
        diffs.append(f"Predicted {predicted.winner}, actual {actual_winner}")
    return {"status": "OK" if not diffs else "Mismatch", "details": diffs}

def load_cases(path: str):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print("Fehler beim Laden:", e)
        return []

def save_cases(path: str, cases: list):
    try:
        json.dump(cases, open(path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("Erfolgreich gespeichert:", path)
    except Exception as e:
        print("Fehler beim Speichern:", e)

# === Hauptskript ===

if __name__ == "__main__":
    INPUT_FILE = "enriched_cases.json"
    OUTPUT_FILE = "baseline_case_predictions.json"
    SYSTEM_PROMPT = (
        """Du bist ein hochspezialisierter juristischer Assistent. Deine Aufgabe ist es, auf Basis des angegebenen Tatbestands eine strukturierte Vorhersage des Tenors im JSON-Format zu erstellen. 
                    Arbeite stets präzise, konsistent und im Rahmen der juristischen Logik. Falls Unsicherheiten bestehen, dokumentiere diese klar. Gib die strukturierte Vorhersage im JSON-Format."""
    )

    cases = load_cases(INPUT_FILE)
    if not cases:
        sys.exit(1)

    cost_usd = estimate_cost(SYSTEM_PROMPT, cases)
    cost_eur = cost_usd * 0.92
    print(f"Geschätzte Kosten für 100 Fälle: {cost_usd:.4f} USD (~{cost_eur:.4f} EUR)")

    if input("Workflow ausführen? (ja/nein): ").strip().lower() != "ja":
        print("Abgebrochen.")
        sys.exit(0)

    correct, total = 0, 0
    for case in tqdm(cases[:100], desc="Verarbeite Fälle"):
        tb = case.get("structured_content", {}).get("tatbestand", "")
        pred = call_openai_api_with_structure(tb)
        actual = case.get("winner")
        case["predicted_tenor"] = pred.dict() if pred else None
        case["comparison"] = compare_tenors(pred, actual)
        if pred and pred.winner == actual:
            correct += 1
        total += 1

    save_cases(OUTPUT_FILE, cases)
    print(f"Fälle: {total}, richtige Gewinner: {correct}, "
          f"Genauigkeit: {correct/total*100:.2f}%")

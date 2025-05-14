import os
import json
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
import openai
import concurrent.futures

# --- Environment Setup ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI()

# --- Models ---
class AnswerWithReason(BaseModel):
    answer: str    # "ja" oder "nein"
    reason: str    # kurze Erläuterung

class QuestionBatchResponse(BaseModel):
    results: List[AnswerWithReason]  # Liste von Objekten mit answer+reason

# --- Batch-Call pro Gesetzesreferenz ---
def call_openai_answer_batch(case_facts: str, questions: List[str], law_ref: str) -> List[Dict[str,str]]:
    """
    Beantworte alle Ja/Nein-Fragen für eine Gesetzesreferenz in einem API-Call
    und liefere pro Frage: {'answer': 'ja'/'nein', 'reason': '…'}.
    """
    system_prompt = (
        f"Du bist ein juristischer Assistent. Für die Gesetzesreferenz {law_ref}:\n"
        "Beantworte jede der folgenden Fragen **nur** mit 'ja' oder 'nein'.\n"
        "Gib außerdem eine kurze Begründung (ein bis zwei Sätze) für jede Antwort an.\n"
        "Antworte im JSON-Format als:\n"
        "{ \"results\": [ {\"answer\":\"ja\",\"reason\":\"…\"}, … ] }"
    )
    user_payload = json.dumps({
        "case_facts": case_facts,
        "questions": questions
    }, ensure_ascii=False)

    try:
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_payload}
            ],
            response_format=QuestionBatchResponse
        )
        # Pydantic parsed: response.choices[0].message.parsed.results
        return [r.dict() for r in response.choices[0].message.parsed.results]
    except Exception as e:
        print(f"Error answering batch for {law_ref}: {e}")
        # Fallback: alle "nein" mit leerer Begründung
        return [{"answer":"nein","reason":""} for _ in questions]


# --- Workflow: load, answer, derive mit Parallelisierung ---
def main():
    # 1. Einlesen
    with open("extracted_laws.json",   "r", encoding="utf-8") as f:
        extracted = json.load(f)
    with open("enriched_cases.json",   "r", encoding="utf-8") as f:
        enriched_cases = json.load(f)[:100]

    # 2. Aufgabenliste: (case_id, law_ref, case_facts, questions)
    tasks: List[Tuple[int,str,str,List[str]]] = []
    for case in extracted:
        cid = case["case_id"]
        enriched = next((c for c in enriched_cases if c.get("id")==cid), {})
        facts = enriched.get("structured_content",{}).get("tatbestand","")
        for law in case["extracted_laws"]:
            qs = [tbm["question"] for tbm in law["tatbestandsmerkmale"]]
            if qs:
                tasks.append((cid, law["law_reference"], facts, qs))

    # 3. Parallel batch calls pro Referenz
    results: Dict[Tuple[int,str], List[Dict[str,str]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futures = {
            ex.submit(call_openai_answer_batch, facts, qs, ref): (cid, ref)
            for cid, ref, facts, qs in tasks
        }
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Answering Batches"):
            cid, ref = futures[fut]
            results[(cid,ref)] = fut.result()

    # 4. Zusammensetzen der Ergebnisse
    output = []
    for case in extracted:
        cid = case["case_id"]
        entry = {"case_id": cid, "laws": []}
        for law in case["extracted_laws"]:
            ref     = law["law_reference"]
            tbms    = law["tatbestandsmerkmale"]
            batches = results.get((cid,ref), [])
            # answers mit reasons zuweisen
            for idx, tbm in enumerate(tbms):
                if idx < len(batches):
                    tbm["answer"] = batches[idx]["answer"]
                    tbm["reason"] = batches[idx]["reason"]
                else:
                    tbm["answer"] = "nein"
                    tbm["reason"] = ""
            # Relevanz prüfen
            relevant = all(tbm["answer"]=="ja" for tbm in tbms)
            rf_art = law.get("rechtsfolge",{}).get("art") if relevant else "Nicht anwendbar"

            entry["laws"].append({
                "law_reference": ref,
                "tatbestandsmerkmale": tbms,
                "relevant": relevant,
                "rechtsfolge": rf_art
            })
        output.append(entry)

    # 5. Speichern
    with open("answered_laws.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print("Fertig! Ergebnisse in 'answered_laws.json'.")

if __name__ == "__main__":
    main()

import os
import json
import tiktoken

# --- Configuration ---
MODEL = "gpt-4o-mini"
PRICING = {
    "input": 0.3,   # $0.03 per 1k input tokens
    "output": 0.6   # $0.06 per 1k output tokens
}

SYSTEM_PROMPT_TEMPLATE = (
    "Du bist ein juristischer Assistent. Beantworte für jede der folgenden Fragen nur mit 'ja' oder 'nein', "
    "gestützt auf den unten stehenden Falltext. Gesetzesreferenz: {law_ref}"
)

# Initialize tiktoken encoder for das Modell
encoder = tiktoken.encoding_for_model(MODEL)

def count_tokens(text: str) -> int:
    """Token-Anzahl für einen Text ermitteln."""
    return len(encoder.encode(text))

def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Kosten basierend auf Token-Zahlen berechnen."""
    cost_in  = (input_tokens  / 1000000) * PRICING["input"]
    cost_out = (output_tokens / 1000000) * PRICING["output"]
    return cost_in + cost_out

def main():
    # 1. Daten einlesen
    with open("extracted_laws.json", "r", encoding="utf-8") as f:
        extracted = json.load(f)
    with open("enriched_cases.json", "r", encoding="utf-8") as f:
        enriched = json.load(f)[:100]

    total_input_tokens = 0
    total_output_tokens = 0
    total_calls = 0

    # 2. Für jede Referenz batch-per-reference Token zählen
    for case in extracted:
        cid = case["case_id"]
        case_enriched = next((c for c in enriched if c["id"] == cid), {})
        case_facts = case_enriched.get("structured_content", {}).get("tatbestand", "")

        for law in case.get("extracted_laws", []):
            ref = law["law_reference"]
            questions = [tbm["question"] for tbm in law.get("tatbestandsmerkmale", [])]
            if not questions:
                continue

            # System-Prompt
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(law_ref=ref)
            input_tokens = count_tokens(system_prompt)

            # User-Payload als JSON-String
            user_payload = json.dumps({
                "case_facts": case_facts,
                "questions": questions
            }, ensure_ascii=False)
            input_tokens += count_tokens(user_payload)

            # Ausgabemenge: eine ja/nein-Antwort pro Frage
            output_tokens = len(questions)

            total_input_tokens  += input_tokens
            total_output_tokens += output_tokens
            total_calls += 1

    # 3. Kosten berechnen
    total_cost = estimate_cost(total_input_tokens, total_output_tokens)

    # 4. Ergebnis ausgeben
    print(f"Batch-per-ref API Calls: {total_calls}")
    print(f"Total Input Tokens:  {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Estimated Cost (USD): ${total_cost:.2f}")

if __name__ == "__main__":
    main()

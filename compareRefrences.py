import json
import re
from typing import List
from tqdm import tqdm

# Function to extract law references
def extract_law_references(text, book_codes, max_length=50):
    book_codes_pattern = r"|".join(map(re.escape, book_codes))
    statute_pattern = rf"(§{{1,2}}.*?\b(?:{book_codes_pattern})\b)"
    matches = re.findall(statute_pattern, text)
    cleaned_matches = [
        " ".join(match.split()) for match in matches if len(match) <= max_length
    ]
    return cleaned_matches

# Function to simplify law references
def simplify_references(references):
    simplified = set()
    for ref in references:
        match = re.search(r"§§?\s*([\d\s,]+).*?(\b\S+$)", ref)
        if match:
            numbers = []
            parts = ref.split(",")
            for part in parts:
                number_match = re.search(r"\b\d+(?=\D|\b)", part)
                if number_match:
                    numbers.append(number_match.group())
            book_code = match.group(2)
            for num in numbers:
                if num.isdigit():
                    simplified.add(f"{num} {book_code}")
    return list(simplified)
def load_book_codes(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Fehler beim Laden der Buch-Codes: {e}")
        return []

# Function to compare references
def compare_references(actual_refs: List[str], hypothetical_refs: List[str]):
    actual_set = set(actual_refs)
    hypothetical_set = set(hypothetical_refs)

    matched = actual_set & hypothetical_set
    only_in_actual = actual_set - hypothetical_set
    only_in_hypothetical = hypothetical_set - actual_set

    return {
        "matched": list(matched),
        "only_in_actual": list(only_in_actual),
        "only_in_hypothetical": list(only_in_hypothetical),
    }

# Main function
if __name__ == "__main__":
    input_file = "hyde_comparison_results_batched_unbiased.json"  # Update with your file path
    output_file = "reference_comparison_results.json"
    book_codes_file="book_codes.json"
    book_codes = load_book_codes(book_codes_file)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
       

        results = []

        for entry in tqdm(data, desc="Processing cases"):
            case_id = entry.get("case_id")
            method = entry.get("method")
            actual_text = entry.get("actual_entscheidungsgruende", "")
            hypothetical_text = entry.get("generated_entscheidungsgruende", "")

            # Extract and simplify references
            actual_refs_long = extract_law_references(actual_text, book_codes)
            hypothetical_refs_long = extract_law_references(hypothetical_text, book_codes)
            actual_refs = simplify_references(actual_refs_long)
            hypothetical_refs = simplify_references(hypothetical_refs_long)

            # Compare references
            comparison = compare_references(actual_refs, hypothetical_refs)

            results.append({
                "case_id": case_id,
                "actual_references": actual_refs,
                "hypothetical_references": hypothetical_refs,
                "comparison": comparison,
                "method": method,
            })

        # Save the results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"Comparison results saved to {output_file}.")

    except Exception as e:
        print(f"Error processing file: {e}")

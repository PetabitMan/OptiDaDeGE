import json
import re
from typing import List
from tqdm import tqdm

# Function to extract law references
def extract_law_references(text, book_codes, max_length=50):
    book_codes_pattern = r"|".join(map(re.escape, book_codes))
    statute_pattern = rf"(§{{1,2}}.*?\b(?:{book_codes_pattern})\b)"
    matches = re.findall(statute_pattern, text)
    cleaned_matches = [" ".join(match.split()) for match in matches if len(match) <= max_length]
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
    # File paths for actual and chain data
    input_file = "hyde_comparison_results_batched_unbiased.json"  # Contains actual_entscheidungsgruende
    chain_file1 = "chain-of-verification_evaluation_results_multi-query.json"
    chain_file2 = "chain-of-verification_evaluation_results_direct.json"  # Contains transformed_hypothetical
    chain_file3 = "chain-of-verification_evaluation_results_least-to-most.json"
    output_file = "reference_comparison_chain_results.json"
    book_codes_file = "book_codes.json"
    
    # Load book codes
    book_codes = load_book_codes(book_codes_file)
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Load all three chain files
        with open(chain_file1, "r", encoding="utf-8") as f:
            chain_data1 = json.load(f)
        with open(chain_file2, "r", encoding="utf-8") as f:
            chain_data2 = json.load(f)
        with open(chain_file3, "r", encoding="utf-8") as f:
            chain_data3 = json.load(f)
            
        # Map methods to their corresponding chain data
        chain_data_mapping = {
            "multi-query": chain_data1,
            "direct": chain_data2,
            "least-to-most": chain_data3,
        }
        
        results = []
        
        for entry in tqdm(data, desc="Processing cases"):
            case_id = entry.get("case_id")
            method = entry.get("method")
            actual_text = entry.get("actual_entscheidungsgruende", "")
            
            # Select the appropriate chain data based on the method
            selected_chain_data = chain_data_mapping.get(method, [])
            chain_entries = [
                item for item in selected_chain_data 
                if item.get("case_id") == case_id and item.get("method") == method
            ]
            if chain_entries:
                transformed_text = chain_entries[0].get("transformed_hypothetical", "")
            else:
                transformed_text = "No transformed_hypothetical found for this case/method."
            
            # Extract and simplify law references from both texts
            actual_refs_long = extract_law_references(actual_text, book_codes)
            transformed_refs_long = extract_law_references(transformed_text, book_codes)
            actual_refs = simplify_references(actual_refs_long)
            transformed_refs = simplify_references(transformed_refs_long)
            
            # Compare the extracted references
            comparison = compare_references(actual_refs, transformed_refs)
            
            results.append({
                "case_id": case_id,
                "actual_references": actual_refs,
                "transformed_references": transformed_refs,
                "comparison": comparison,
                "method": method,
            })
        
        # Save the comparison results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"Comparison results saved to {output_file}.")
    
    except Exception as e:
        print(f"Error processing file: {e}")

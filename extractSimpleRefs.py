import json

def extract_simple_refs(input_file, output_file, num_cases=100):
    # Load the enriched cases from the input file
    with open(input_file, 'r', encoding='utf-8') as infile:
        cases = json.load(infile)
    
    # Initialize an empty set to store unique simple_refs
    simple_refs = set()

    # Process only the first num_cases cases
    for case in cases[:num_cases]:
        refs = case.get("simple_refs")
        if refs:
            if isinstance(refs, list):
                simple_refs.update(refs)
            else:
                simple_refs.add(refs)
    
    # Convert the set to a list to ensure JSON serializability
    simple_refs_list = list(simple_refs)
    
    # Save the resulting list to the output JSON file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(simple_refs_list, outfile, ensure_ascii=False, indent=4)
    
    print(f"Extracted {len(simple_refs_list)} unique simple_refs from the first {num_cases} cases.")

if __name__ == "__main__":
    # Input file containing the enriched cases (assumed JSON format)
    input_file = "enriched_cases.json"
    # Output file where the unique simple_refs will be saved
    output_file = "simple_refs.json"
    extract_simple_refs(input_file, output_file)

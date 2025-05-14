import json

def find_case_references(input_file_10, input_file_90):
    """
    Extracts case references from the 10% dataset, searches them in the 90% dataset,
    and returns the matching IDs.

    :param input_file_10: Path to the JSON file containing 10% of the cases.
    :param input_file_90: Path to the JSON file containing 90% of the cases.
    :return: A dictionary mapping references to their corresponding IDs.
    """
    # Load the data files
    with open(input_file_10, 'r', encoding='utf-8') as f:
        cases_10_percent = json.load(f)
    with open(input_file_90, 'r', encoding='utf-8') as f:
        cases_90_percent = json.load(f)

    # Create a lookup dictionary for the 90% cases using file number as the key
    case_lookup = {case.get("file_number"): case.get("id") for case in cases_90_percent if "file_number" in case}

    # Extract references and find matches
    reference_id_map = {}
    for case in cases_10_percent:
        case_references = case.get("case_references", [])  # Assumes references are in a list under "case_references"
        for reference in case_references:
            file_number = reference
            if file_number and file_number in case_lookup:
                reference_id_map.setdefault(case["id"], []).append(case_lookup[file_number])

    return reference_id_map


def add_relevant_cases(input_file_10, input_file_90, output_file):
    """
    Adds relevant_cases field to each case in the 10% dataset and saves it to a new file.

    :param input_file_10: Path to the JSON file containing 10% of the cases.
    :param input_file_90: Path to the JSON file containing 90% of the cases.
    :param output_file: Path to save the updated 10% dataset.
    """
    with open(input_file_10, 'r', encoding='utf-8') as f:
        cases_10_percent = json.load(f)

    # Find the references and their corresponding IDs
    reference_id_map = find_case_references(input_file_10, input_file_90)

    # Add the relevant_cases field to each case
    for case in cases_10_percent:
        case_id = case.get("id")
        case["relevant_cases"] = reference_id_map.get(case_id, [])

    # Save the updated dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cases_10_percent, f, indent=4, ensure_ascii=False)

    print(f"Updated 10% dataset saved to {output_file}")


# Example usage
input_file_10 = "enriched_cases.json"
input_file_90 = "training_cases.json"
output_file = "enriched_cases.json"

add_relevant_cases(input_file_10, input_file_90, output_file)

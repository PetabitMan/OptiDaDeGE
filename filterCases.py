import ijson
import json

def filter_large_json(input_file, output_file, criteria_func):
    """
    Filters a large JSON array file incrementally and writes the filtered results to a new file.

    :param input_file: Path to the input JSON file.
    :param output_file: Path to the output JSON file.
    :param criteria_func: Function to filter cases based on specific criteria.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write("[\n")  # Start JSON array in output file
            
            # Use ijson to parse the JSON array incrementally
            items = ijson.items(infile, 'item')  # 'item' parses each element of the JSON array
            
            first = True
            for item in items:
                if criteria_func(item):  # Apply the filtering criteria
                    if not first:
                        outfile.write(",\n")
                    json.dump(item, outfile, ensure_ascii=False)
                    first = False

            outfile.write("\n]")  # End JSON array
        print(f"Filtered data saved to {output_file}")
    
    except Exception as e:
        print(f"Error processing file: {e}")

# Define the filtering criteria
def filter_criteria(case):
    """
    Filtering criteria for cases. Only include cases where all required fields are present.
    """
    structured_content = case.get('structured_content', {})
    return (
        structured_content.get('tenor') and 
        structured_content.get('tatbestand') and 
        structured_content.get('entscheidungsgruende')
    )

# Main script
if __name__ == "__main__":
    input_file = 'court_cases_structured_no_content_test.json'
    output_file = 'filtered_cases.json'
    filter_large_json(input_file, output_file, filter_criteria)

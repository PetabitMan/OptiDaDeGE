import json
import random

def split_filtered_file(input_file, output_10_file, output_90_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)  # Load the filtered JSON array
        random.shuffle(data)  # Shuffle the cases
        
        # Split the data into 10% and 90%
        split_index = len(data) // 10
        cases_10_percent = data[:split_index]
        cases_90_percent = data[split_index:]

        # Write the 10% and 90% subsets to separate files
        with open(output_10_file, 'w', encoding='utf-8') as out10:
            json.dump(cases_10_percent, out10, ensure_ascii=False, indent=4)
        with open(output_90_file, 'w', encoding='utf-8') as out90:
            json.dump(cases_90_percent, out90, ensure_ascii=False, indent=4)
        
        print(f"10% subset saved to {output_10_file} ({len(cases_10_percent)} cases)")
        print(f"90% subset saved to {output_90_file} ({len(cases_90_percent)} cases)")

# Main script
if __name__ == "__main__":
    input_file = 'filtered_cases.json'
    output_10_file = 'filtered_cases_10_percent.json'
    output_90_file = 'filtered_cases_90_percent.json'
    
    split_filtered_file(input_file, output_10_file, output_90_file)

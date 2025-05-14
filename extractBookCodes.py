import json

# Function to extract and extend book codes from a JSONL file
def extract_book_codes(jsonl_file):
    book_codes = set()
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                if 'code' in data:
                    code = data['code']
                    book_codes.add(code)
                    # If the code contains numbers, add a version without numbers
                    if any(char.isdigit() for char in code):
                        code_without_numbers = ''.join(filter(lambda x: not x.isdigit(), code))
                        code_without_numbers_and_spaces = code_without_numbers.replace(" ", "")

                        book_codes.add(code_without_numbers_and_spaces)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return list(book_codes)


# Function to save the book codes to a JSON file
def save_book_codes(book_codes, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(book_codes, file, ensure_ascii=False, indent=4)
    print(f"Book codes saved to {output_file}")

# Example usage
if __name__ == "__main__":
    jsonl_file = "law_books.jsonl"  # Replace with your actual file path
    output_file = "book_codes.json"  # Replace with your desired output file path

    book_codes = extract_book_codes(jsonl_file)
    print(f"Extracted Book Codes: {book_codes}")

    save_book_codes(book_codes, output_file)

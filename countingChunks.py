import json
from tqdm import tqdm
import tiktoken

def count_tokens(text, model="text-embedding-3-small"):
    """
    Counts the number of tokens in a given text using the specified model tokenizer.
    """
    tokenizer = tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(text))

def split_text_on_newline_with_token_limit(text, token_limit=1000, model="text-embedding-3-small"):
    """
    Splits the text into chunks based on '--NEWLINE--' and merges smaller chunks 
    until the total token count exceeds the specified limit.
    """
    tokenizer = tiktoken.encoding_for_model(model)
    chunks = text.split("--NEWLINE--")
    merged_chunks = []
    current_chunk = []
    current_token_count = 0

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        chunk_token_count = len(tokenizer.encode(chunk))
        
        if current_token_count + chunk_token_count > token_limit and current_chunk:
            merged_chunks.append(" ".join(current_chunk))
            current_chunk = [chunk]
            current_token_count = chunk_token_count
        else:
            current_chunk.append(chunk)
            current_token_count += chunk_token_count

    if current_chunk:
        merged_chunks.append(" ".join(current_chunk))

    return merged_chunks

def count_documents(file_path):
    """
    Reads a JSON file containing training cases, splits text into chunks,
    counts the total documents, and calculates the total token count.

    :param file_path: Path to the JSON file containing training cases.
    :return: Total number of documents and total token count.
    """
    try:
        print(f"Loading cases from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            cases = json.load(f)

        total_documents = 0
        total_tokens = 0

        print("Processing cases...")
        for case in tqdm(cases, desc="Processing cases"):
            content = case.get('structured_content', {}).get('entscheidungsgruende', '').strip()
            if not content:
                continue  # Skip cases with empty Entscheidungsgründe

            chunks = split_text_on_newline_with_token_limit(content, token_limit=500)
            total_documents += len(chunks)

            # Calculate tokens for each chunk and add to total_tokens
            for chunk in chunks:
                total_tokens += count_tokens(chunk)

        print(f"Total documents: {total_documents}")
        print(f"Total token count: {total_tokens}")
        return total_documents, total_tokens

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    training_cases_path = 'training_cases.json'  # Replace with your actual file path
    count_documents(training_cases_path)

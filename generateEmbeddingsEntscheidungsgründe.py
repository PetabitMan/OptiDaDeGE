import os
import json
import time
import tiktoken
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from tqdm import tqdm
from threading import Semaphore
import requests

# Load environment variables
load_dotenv()
def count_tokens(text, model="text-embedding-3-small"):
    """
    Count the number of tokens in a given text using a specified model.
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
class RateLimiter:
    def __init__(self, max_requests, period):
        """
        Initialize the rate limiter.
        :param max_requests: Maximum allowed requests in the given period.
        :param period: Time period in seconds.
        """
        self.max_requests = max_requests
        self.period = period
        self.requests = []

    def wait_for_slot(self):
        """
        Wait for a slot to become available in the rate limiter.
        """
        current_time = time.time()
        # Remove expired requests
        self.requests = [req for req in self.requests if req > current_time - self.period]

        if len(self.requests) >= self.max_requests:
            # Wait until the oldest request expires
            wait_time = self.period - (current_time - self.requests[0])
            print(f"Rate limit reached. Waiting for {wait_time:.2f} seconds...")
            time.sleep(wait_time)

        # Register a new request
        self.requests.append(current_time)

def exponential_backoff(func, *args, max_retries=5, **kwargs):
    """
    Calls a function with exponential backoff in case of rate limit errors.
    """
    backoff_time = 1  # Start with 1 second

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)  # Call the function
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                print(f"Rate limit error: {e}. Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 2  # Double the backoff time
            else: 
                raise  # Re-raise other exceptions
    raise RuntimeError("Max retries exceeded due to rate limiting.")

def get_retry_after(headers):
    """
    Extracts the 'Retry-After' value from response headers, if available.
    """
    if 'Retry-After' in headers:
        return float(headers['Retry-After'])
    elif 'X-RateLimit-Reset' in headers:  # OpenAI-specific header
        reset_time = float(headers['X-RateLimit-Reset'])
        current_time = time.time()
        return max(0, reset_time - current_time)
    return None

def call_with_dynamic_wait(func, *args, **kwargs):
    """
    Calls a function and dynamically adjusts wait time based on API response headers.
    """
    while True:
        try:
            return func(*args, **kwargs)  # Call the API function
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                retry_after = get_retry_after(e.response.headers)
                if retry_after:
                    print(f"Rate limit hit. Retrying after {retry_after} seconds...")
                    time.sleep(retry_after)
                else:
                    print("Rate limit hit. Retrying with default wait time...")
                    time.sleep(60)  # Default wait
            else:
                raise

def generate_embeddings(cases, model="text-embedding-3-small", batch_size=150):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")

    embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)
    documents = []
    rate_limiter = RateLimiter(max_requests=1000, period=60)  # 2500 requests per minute
    semaphore = Semaphore(20)  # Limit to 5 concurrent API calls

    # Prepare documents
    for case in tqdm(cases, desc="Preparing Documents"):
        content = case['structured_content'].get('entscheidungsgruende', '').strip()
        if not content:
            print(f"Skipping case ID {case.get('id')} - Empty Entscheidungsgründe.")
            continue

        court_info = case.get('court', {})
        jurisdiction = court_info.get('jurisdiction', 'Unknown')
        level_of_appeal = court_info.get('level_of_appeal', 'Unknown')

        chunks = split_text_on_newline_with_token_limit(content)
        for i, chunk in enumerate(chunks):
            if not chunk.strip():  # Check if the chunk is empty or whitespace
                continue
            if chunk == "gründe:":  # Skip specific chunk
                continue
            metadata = {
                "id": case.get("id"),
                "chunk_id": i,
                "jurisdiction": jurisdiction,
                "level_of_appeal": level_of_appeal,
            }
            documents.append(Document(page_content=chunk, metadata=metadata))

    vectorstore = None
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} with {len(batch)} documents...")

        def process_batch():
            # Call FAISS with rate limiting and backoff
            rate_limiter.wait_for_slot()
            return exponential_backoff(FAISS.from_documents, batch, embeddings)

        try:
            with semaphore:
                batch_embeddings = process_batch()

            # Merge the embeddings into the vectorstore
            if vectorstore is None:
                vectorstore = batch_embeddings
            else:
                vectorstore.merge_from(batch_embeddings)
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            continue

    if vectorstore:
        print(f"Generated embeddings for {len(documents)} documents.")
    else:
        print("No documents to embed. Exiting.")
    return vectorstore

if __name__ == "__main__":
    try:
        print("loading cases")
        with open('training_cases.json', 'r', encoding='utf-8') as f:
            cases = json.load(f)

        vectorstore = generate_embeddings(cases, model="text-embedding-3-small", batch_size=100)
        if vectorstore:
            vectorstore.save_local("faiss_legal_index_entscheidungsgruende")
            print("FAISS vector store saved successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")


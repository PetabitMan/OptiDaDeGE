import os
import json
import time
import tiktoken
import threading
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from tqdm import tqdm
import requests
import concurrent.futures
import faiss  # Falls du direkt mit faiss arbeiten möchtest

# Load environment variables
load_dotenv()

# Globaler Speicherort für den FAISS-Index (als Ordner)
INDEX_PATH = "faiss_legal_index_entscheidungsgruende"

def count_tokens(text, model="text-embedding-3-small"):
    """
    Zählt die Tokens in einem Text für ein gegebenes Modell.
    """
    tokenizer = tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(text))

def split_text_on_newline_with_token_limit(text, token_limit=1000, model="text-embedding-3-small"):
    """
    Teilt den Text anhand von "--NEWLINE--" und fügt Teilsätze zusammen,
    bis der Token-Limit erreicht ist.
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

# --------------------------
# RATE LIMITING CLASSES
# --------------------------

class RateLimiter:
    """
    Beschränkt die Anzahl der API-Aufrufe pro Zeitraum.
    (Hier sind maximal 5000 Requests pro 60 Sekunden erlaubt.)
    """
    def __init__(self, max_requests, period):
        self.max_requests = max_requests
        self.period = period
        self.requests = []
        self.lock = threading.Lock()

    def wait_for_slot(self):
        while True:
            with self.lock:
                current_time = time.time()
                # Entferne abgelaufene Requests
                self.requests = [req for req in self.requests if req > current_time - self.period]
                if len(self.requests) < self.max_requests:
                    self.requests.append(current_time)
                    return
                else:
                    wait_time = self.period - (current_time - self.requests[0])
            print(f"Rate limit (requests) erreicht. Warte {wait_time:.2f} Sekunden...")
            time.sleep(wait_time)

def exponential_backoff(func, *args, max_retries=5, **kwargs):
    """
    Ruft eine Funktion mit exponentiellem Backoff bei Rate-Limit-Fehlern auf.
    """
    backoff_time = 1  # Startzeit 1 Sekunde
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                print(f"Rate limit error: {e}. Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 2
            else:
                raise
    raise RuntimeError("Maximale Anzahl von Retries wegen Rate Limiting überschritten.")

class TokenBucket:
    """
    Ein Token-Bucket, um die Token-Nutzung pro Zeitraum zu begrenzen.
    (Kapazität: 1.000.000 Tokens pro Minute)
    """
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # Tokens pro Sekunde
        self.last_time = time.time()
        self.lock = threading.Lock()

    def wait_for_tokens(self, tokens_needed):
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= tokens_needed:
                    self.tokens -= tokens_needed
                    return
                else:
                    needed = tokens_needed - self.tokens
                    wait_time = needed / self.refill_rate
            print(f"Token bucket: Nicht genügend Tokens, warte {wait_time:.2f} Sekunden...")
            time.sleep(wait_time)

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_time = now

# --------------------------
# FAISS CHECKPOINT FUNCTIONS
# --------------------------

def load_faiss_index(embeddings):
    """
    Lädt einen bestehenden FAISS-Index, falls vorhanden, ansonsten None.
    """
    if os.path.exists(INDEX_PATH):
        try:
            print("Lade bestehenden FAISS-Index ...")
            vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print(f"FAISS geladen: {vectorstore.index.ntotal} Vektoren vorhanden.")
            return vectorstore
        except Exception as e:
            print(f"Fehler beim Laden des FAISS-Index: {e}")
            return None
    else:
        print("Kein bestehender FAISS-Index gefunden. Starte neu.")
        return None

def save_faiss_index(vectorstore):
    """
    Speichert den FAISS-Index (Checkpoint) lokal.
    """
    if vectorstore:
        vectorstore.save_local(INDEX_PATH)
        print(f"FAISS gespeichert: {vectorstore.index.ntotal} Vektoren.")

# --------------------------
# MAIN EMBEDDING FUNCTION
# --------------------------

def generate_embeddings(cases, model="text-embedding-3-small", batch_size=100, max_workers=5):
    """
    Verarbeitet die Fälle und baut einen FAISS-Vektorstore auf.
    Dabei werden Token- und Request-Limits eingehalten sowie Checkpoints gesetzt.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY ist nicht gesetzt in der Umgebung.")

    # Erzeuge das Embeddings-Objekt
    embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)
    
    # Lade eventuell bereits vorhandenen FAISS-Index (Checkpoint)
    vectorstore = load_faiss_index(embeddings)

    # Bereite die Dokumente vor
    documents = []
    for case in tqdm(cases, desc="Preparing Documents"):
        content = case['structured_content'].get('entscheidungsgruende', '').strip()
        if not content:
            print(f"Skipping case ID {case.get('id')} - Empty Entscheidungsgründe.")
            continue

        court_info = case.get('court', {})
        jurisdiction = court_info.get('jurisdiction', 'Unknown')
        level_of_appeal = court_info.get('level_of_appeal', 'Unknown')

        # Nutze den Splitter (alternativ kannst du auch split_text_on_newline_with_token_limit verwenden)
        chunks = split_text_on_newline_with_token_limit(content)
        for i, chunk in enumerate(chunks):
            if not chunk.strip() or chunk == "gründe:":
                continue
            metadata = {
                "id": case.get("id"),
                "chunk_id": i,
                "jurisdiction": jurisdiction,
                "level_of_appeal": level_of_appeal,
            }
            documents.append(Document(page_content=chunk, metadata=metadata))

    # Teile die Dokumente in Batches
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    total_batches = len(batches)
    print(f"Total batches to process: {total_batches}")

    # Berechne, wie viele Batches bereits verarbeitet wurden (angenommen: 1 Dokument = 1 Vektor)
    processed_batches = 0
    if vectorstore:
        processed_batches = vectorstore.index.ntotal // batch_size
        print(f"Bereits verarbeitete Batches: {processed_batches}")

    # Erstelle einen Token Bucket (1.000.000 Tokens pro Minute)
    token_bucket = TokenBucket(capacity=1000000, refill_rate=1000000/60)
    rate_limiter = RateLimiter(max_requests=5000, period=60)

    # Funktion zur Verarbeitung eines einzelnen Batches
    def process_batch(batch, batch_index):
        tokens_needed = sum(count_tokens(doc.page_content, model=model) for doc in batch)
        token_bucket.wait_for_tokens(tokens_needed)
        rate_limiter.wait_for_slot()
        print(f"Processing batch {batch_index+1}/{total_batches} with {len(batch)} documents ({tokens_needed} tokens)...")
        return exponential_backoff(FAISS.from_documents, batch, embeddings)

    # Verarbeite die Batches – überspringe bereits verarbeitete Batches
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, batch in enumerate(batches):
            if idx < processed_batches:
                print(f"Skipping batch {idx+1} (already processed).")
                continue
            future = executor.submit(process_batch, batch, idx)
            futures[future] = idx

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing batches"):
            batch_index = futures[future]
            try:
                batch_vectorstore = future.result()
                if vectorstore is None:
                    vectorstore = batch_vectorstore
                else:
                    vectorstore.merge_from(batch_vectorstore)
                # Speichere nach jedem verarbeiteten Batch (Checkpoint)
                save_faiss_index(vectorstore)
            except Exception as e:
                print(f"Error processing batch {batch_index+1}: {e}")

    if vectorstore:
        print(f"Generated embeddings for {len(documents)} documents.")
    else:
        print("No documents to embed. Exiting.")
    return vectorstore

# --------------------------
# MAIN ENTRY POINT
# --------------------------

if __name__ == "__main__":
    try:
        print("Loading cases...")
        with open('training_cases.json', 'r', encoding='utf-8') as f:
            cases = json.load(f)

        # Passe batch_size und max_workers bei Bedarf an.
        vectorstore = generate_embeddings(cases, model="text-embedding-3-small", batch_size=100, max_workers=5)
        if vectorstore:
            # Finaler Save (falls noch nicht geschehen)
            vectorstore.save_local(INDEX_PATH)
            print("FAISS vector store saved successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

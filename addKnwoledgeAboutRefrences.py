import json
import re
import html
from typing import List, Dict
from pydantic import BaseModel, Field
from tqdm import tqdm
import os
from dotenv import load_dotenv
import asyncio

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ------------------------------------------------------------------
# Load environment variables and initialize FAISS vector store
# ------------------------------------------------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Specify the correct model used during FAISS creation
    openai_api_key=openai_api_key
)
vectorstore = FAISS.load_local(
    "faiss_legal_index_tatbestand", 
    embeddings,
    allow_dangerous_deserialization=True
)

# ------------------------------------------------------------------
# Define structured output for law application results
# ------------------------------------------------------------------
class LawApplication(BaseModel):
    Applies: str = Field(..., description="Explanation of why the law does or does not apply")
    does_apply: bool = Field(..., description="Whether the law applies to the tatbestand")

# ------------------------------------------------------------------
# Initialize ChatOpenAI and create a structured LLM for law application
# ------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
structured_llm_law = llm.with_structured_output(LawApplication)

print("loading laws...")
with open("laws.jsonl", "r", encoding="utf-8") as f:
    laws = [json.loads(line) for line in f]
with open("law_books.jsonl", "r", encoding="utf-8") as f:
    law_books_list = [json.loads(line) for line in f]
# Erzeuge ein Mapping: Buch-Code → Buch-Objekt
law_books = {entry["code"]: entry for entry in law_books_list if "code" in entry}

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Clean legal text by decoding HTML entities, removing HTML tags, and extra spaces."""
    text = html.unescape(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def normalize_slug(slug: str) -> str:
    """
    Normalize legal slugs by:
    - Keeping digits and lowercase letters.
    - Removing spaces, paragraphs signs (§), "Abs.", "Satz", commas, dots, etc.
    """
    slug = slug.lower()
    slug = slug.replace("§", "").replace("abs.", "").replace("satz", "")
    slug = re.sub(r"[^\w]", "", slug)  # Keep only word characters (letters+numbers)
    slug = slug.strip()
    return slug

def search_law(book: str, slug: str, laws: List[dict], law_books: dict) -> str:
    """
    Search for a specific law text using book and slug.
    Uses exact code match first, then fuzzy title search.
    """
    manual_book_mapping = {
        "GG": 2215,
        "BVerfG": 1371,
        "AufenthG": 1175,
        "AO": 1150,
        "ASYLVFG": 1167,
        "BGB": 1280,
        "STVO": 2024,
        "SGB": 1972,
        # "VwGO": 1190,
        "ZPO": 2199,
        "STVZO": 2026,
        "ROG": 1906,
        # "StGB": 1230,
        # "GKG": 1384,
        # "VwVfG": 1189
    }

    normalized_book = book.upper()
    book_id = manual_book_mapping.get(normalized_book)

    # First try: Direct key in law_books
    if not book_id:
        book_entry = law_books.get(book)
        if book_entry:
            book_id = book_entry["id"]

    # Second try: Fuzzy title matching
    if not book_id:
        for code, entry in law_books.items():
            if normalized_book in entry.get("title", "").upper() or normalized_book in code.upper():
                book_id = entry["id"]
                break

    if not book_id:
        print(f"⚠️ Book {book} not found.")
        return ""

    # Now search the law
    target_slug = normalize_slug(slug)

    for law in laws:
        if law.get("book") == book_id:
            law_slug = law.get("slug", "")
            law_slug_normalized = normalize_slug(law_slug)

            if law_slug_normalized == target_slug:
                return clean_text(law.get("content", ""))

    print(f"⚠️ Law {slug} {book} not found in book ID {book_id}.")
    return ""

def check_law_application(tatbestand: str, law_ref: str, law_text: str) -> LawApplication:
    """
    Use a structured LLM call to check whether the provided law applies to the tatbestand.
    The LLM returns a JSON that fits the LawApplication schema.
    """
    prompt = f"""
        Bitte lesen Sie den folgenden Tatbestand und analysieren Sie detailliert, ob der untenstehende Gesetzestext auf diesen Fall anwendbar ist.

        ---------------------------------------------------------
        Tatbestand:
        {tatbestand}
        ---------------------------------------------------------

        Gesetzestext für die Referenz "{law_ref}":
        ---------------------------------------------------------
        {law_text}
        ---------------------------------------------------------

        Aufgabe:
        1. Überprüfen Sie, ob der Gesetzestext relevante Regelungen enthält, die im Kontext des oben dargestellten Tatbestands zur Anwendung kommen.
        2. Erklären Sie Schritt für Schritt (Chain-of-Thought), welche Aspekte des Tatbestands mit den Regelungen des Gesetzes übereinstimmen oder im Widerspruch stehen.
        3. Fassen Sie abschließend zusammen, ob der Gesetzestext als anwendbar (true) oder nicht anwendbar (false) einzustufen ist, und begründen Sie Ihre Entscheidung präzise.

        Bitte geben Sie Ihre Antwort ausschließlich im folgenden JSON-Format zurück (ohne zusätzliche Kommentare):

        {{
            "Applies": "Ihre ausführliche juristische Begründung, warum der Gesetzestext anwendbar ist oder nicht.",
            "does_apply": true/false
        }}
            """
    prompt_v1 = f"""
        Gegeben den folgenden Tatbestand:
        {tatbestand}

        Und den folgenden Gesetzestext für die Referenz "{law_ref}":
        {law_text}

        Analysieren Sie bitte, ob dieser Gesetzestext auf den Tatbestand anwendbar ist. Erklären Sie, warum oder warum nicht.
        Geben Sie Ihre Antwort im folgenden JSON-Format zurück (ohne zusätzliche Kommentare):
        {{
            "Applies": "Ihre Erklärung hier",
            "does_apply": true/false
        }}
            """
    try:
        response = structured_llm_law.invoke(prompt, temperature=0)
        return response
    except Exception as e:
        print(f"LLM error for law reference '{law_ref}': {e}")
        return LawApplication(Applies="LLM call failed", does_apply=False)

async def async_check_law_application(tatbestand: str, law_ref: str, law_text: str) -> LawApplication:
    """
    Asynchronous wrapper around check_law_application using run_in_executor.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, check_law_application, tatbestand, law_ref, law_text)

def find_similar_case_docs(tatbestand: str, query_jurisdiction: str, k: int = 5):
    """
    Use the FAISS vector store to perform a similarity search for the given tatbestand.
    Filter candidate documents by matching jurisdiction (if provided) and return the top k.
    """
    candidate_docs = vectorstore.similarity_search(tatbestand, k=k * 3)
    if query_jurisdiction:
        retrieved_docs = [
            doc for doc in candidate_docs
            if doc.metadata 
            and str(doc.metadata.get("jurisdiction", "")).lower() == query_jurisdiction.lower()
        ]
    else:
        retrieved_docs = candidate_docs
    return retrieved_docs[:k]

# ------------------------------------------------------------------
# Asynchronous main processing function
# ------------------------------------------------------------------
async def async_process_law_applications(input_file: str, training_cases_file: str, output_file: str):
    """
    For each of the first 100 enriched cases:
      - Use FAISS similarity search (with jurisdiction filtering) to retrieve similar cases.
      - Extract the candidate IDs from the retrieved documents.
      - For each candidate ID, look up the corresponding training case from training_cases
        to retrieve its "simple_refs" field.
      - Aggregate all unique law references, look up their law texts, and call the LLM to check applicability concurrently.
      - Save the results for each case in a JSON file.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        enriched_cases = json.load(f)
    
    with open(training_cases_file, "r", encoding="utf-8") as f:
        training_cases_list = json.load(f)
    training_cases = {case["id"]: case for case in training_cases_list if "id" in case}
    
    results = []
    
    for case in tqdm(enriched_cases[:100], desc="Processing Cases"):
        case_id = case.get("id")
        tatbestand = case.get("structured_content", {}).get("tatbestand", "")
        if not tatbestand:
            continue
        
        jurisdiction = case.get("court", {}).get("jurisdiction", "")
        similar_docs = find_similar_case_docs(tatbestand, jurisdiction, k=5)
        candidate_ids = [doc.metadata.get("id") for doc in similar_docs if doc.metadata.get("id")]
        
        law_refs = set()
        for cid in candidate_ids:
            training_case = training_cases.get(cid)
            if training_case:
                simple_refs = training_case.get("simple_refs", [])
                for ref in simple_refs:
                    law_refs.add(ref)
        law_refs = list(law_refs)
        
        # Launch law application checks concurrently
        tasks = []
        for ref in law_refs:
            parts = ref.split()
            if len(parts) < 2:
                continue
            slug = parts[0]       # e.g., "114"
            book = parts[1]       # e.g., "VwGO"
            law_text = search_law(book, slug, laws, law_books)
            tasks.append(async_check_law_application(tatbestand, ref, law_text))
        law_applications_list = await asyncio.gather(*tasks)
        
        law_results = {ref: result.dict() for ref, result in zip(law_refs, law_applications_list)}
        results.append({
            "case_id": case_id,
            "law_applications": law_results
        })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {output_file}")

# ------------------------------------------------------------------
# Example call
# ------------------------------------------------------------------
if __name__ == "__main__":
    input_file = "enriched_cases.json"         # Enriched cases to process
    training_cases_file = "training_cases.json"  # Training cases containing "simple_refs"
    output_file = "law_application_results.json" # Output file for the results
    asyncio.run(async_process_law_applications(input_file, training_cases_file, output_file))

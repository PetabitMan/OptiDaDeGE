from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
import json
import tiktoken
import time
from tqdm import tqdm

# Load the .env file
load_dotenv()

def count_tokens(text, model="text-embedding-3-small"):
    """
    Count the number of tokens in a given text using a specified model.
    """
    tokenizer = tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(text))

def split_text(text, model="text-embedding-3-small", max_tokens=8000):
    """
    Splits text into chunks that fit within the token limit.
    """
    tokenizer = tiktoken.encoding_for_model(model)
    tokens = tokenizer.encode(text)
    chunks = []
    
    while tokens:
        chunk_tokens = tokens[:max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        tokens = tokens[max_tokens:]
    
    return chunks

def generate_embeddings(cases, model="text-embedding-3-small", max_tokens=5000, batch_size=150):
    """
    Generate embeddings for a list of cases in batches to handle rate limits.
    """
    # Retrieve OpenAI API key from the environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")

    # Initialize the embedding model
    embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)
    documents = []

    # Prepare the documents for embedding
    for case in tqdm(cases, desc="Proccessing Cases"):
        content = case['structured_content'].get('tatbestand', '').strip()
        if not content:
            print(f"Skipping case ID {case.get('id')} - Empty Tatbestand.")
            continue
        court_info = case.get('court', {})
        jurisdiction = court_info.get('jurisdiction', 'Unknown')
        level_of_appeal = court_info.get('level_of_appeal', 'Unknown')

        chunks = split_text(content, model=model, max_tokens=max_tokens)
        for i, chunk in enumerate(chunks):
            metadata = {
                            "id": case.get("id"),
                            "chunk_id": i,
                            "jurisdiction": jurisdiction,
                            "level_of_appeal": level_of_appeal,
                        }     
            documents.append(Document(page_content=chunk, metadata=metadata))

    # Process the documents in batches
    vectorstore = None
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} with {len(batch)} of documents... of {len(documents)/batch_size} batches")

        # Generate embeddings for the batch
        batch_embeddings = FAISS.from_documents(batch, embeddings)

        # Merge batches into a single vector store
        if vectorstore is None:
            vectorstore = batch_embeddings
        else:
            vectorstore.merge_from(batch_embeddings)

        # Delay to respect rate limits
        time.sleep(60)  # Wait 60 seconds between batches

    if vectorstore:
        print(f"Generated embeddings for {len(documents)} documents.")
    else:
        print("No documents to embed. Exiting.")
    return vectorstore

if __name__ == "__main__":
    # Load the filtered legal cases
    with open('training_cases.json', 'r', encoding='utf-8') as f:
        cases = json.load(f)

    # Generate embeddings and save the vector store
    vectorstore = generate_embeddings(cases, model="text-embedding-3-small", max_tokens=5000, batch_size=200)
    if vectorstore:
        vectorstore.save_local("faiss_legal_index_tatbestand")
        print("FAISS vector store saved successfully!")

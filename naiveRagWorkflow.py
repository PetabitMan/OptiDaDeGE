import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional, Union
from tqdm import tqdm
import logging
import re
import asyncio
import pickle
import tiktoken
import heapq

from UtilityFunctions.llmCritique import filter_and_summarize_batch

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the FAISS vector store and embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Specify the correct model used during FAISS creation
    openai_api_key=openai_api_key
)
vectorstore = FAISS.load_local(
    "faiss_legal_index_tatbestand", 
    embeddings,
    allow_dangerous_deserialization=True
)
# Load BM25 index and metadata for tatbestand
bm25_tatbestand_index_file = os.path.join("bm25_indices", "bm25_tatbestand_index.pkl")
bm25_tatbestand_metadata_file = os.path.join("bm25_indices", "bm25_tatbestand_metadata.json")
print(f"Loading BM25 index for tatbestand from {bm25_tatbestand_index_file} ...")
with open(bm25_tatbestand_index_file, "rb") as pkl_in:
    bm25_tatbestand_index = pickle.load(pkl_in)
with open(bm25_tatbestand_metadata_file, "r", encoding="utf-8") as meta_in:
    bm25_tatbestand_metadata = json.load(meta_in)
# New tokenization function: simple word-level tokenization.
def tokenize_text(text):
    # Lowercase the text and extract word tokens (alphanumeric + underscore).
    tokens = re.findall(r'\w+', text.lower())
    return tokens
# Generic BM25 search function for a given section
def get_top_bm25_matches_for_section(query, bm25_index, bm25_metadata, section_name, k=5):
    query_tokens = tokenize_text(query)
    scores = bm25_index.get_scores(query_tokens)
    top_indices = heapq.nlargest(k, range(len(scores)), key=lambda i: scores[i])
    #top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    matches = []
    for idx in top_indices:
        meta = bm25_metadata[idx]
        matches.append({
            # Use the section name as the key for clarity in the prompt later
            section_name: meta["content"],
            "id": meta["id"],
            "score": scores[idx]
        })
    return matches
# Pydantic-Modell für das strukturierte JSON-Format
# class CostBorneBy(BaseModel):
#     prozentzahl_kläger: Union[float, None]
#     prozentzahl_angeklagte: Union[float, None]

class PredictedTenor(BaseModel):
    winner: str = Field(..., description="Der vorhergesagte Gewinner ('Kläger' oder 'Beklagter').")
    # costs_borne_by: CostBorneBy = Field(
    #     ..., description="Ein Dictionary mit Prozentangaben für Kläger und Beklagter."
    # )
    # amount_to_be_paid: Optional[float] = Field(
    #     None, description="Der zu zahlende Betrag, falls vorhanden."
    # )
    # interest: Optional[str] = Field(None, description="'ja' oder 'nein', ob Zinsen anfallen.")
    confidence_notes: str = Field(..., description="Hinweise zur Vorhersagegenauigkeit oder Sicherheit.")

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
# Füge das strukturierte Ausgabeformat hinzu
structured_llm = llm.with_structured_output(PredictedTenor)


# Normalize winner names
def normalize_winner_name(name):
    mapping = {
        "Angeklagte": "Defendant",
        "Angeklagter": "Defendant",
        "Beklagte": "Defendant",
        "Beklagter": "Defendant",
        "Kläger": "Plaintiff",
        "Klägerin": "Plaintiff",
        "Plaintiff": "Plaintiff",
        "Defendant": "Defendant"
    }
    return mapping.get(name, name)

# Load the additional case data
with open("training_cases_with_summaries_completed.json", "r", encoding="utf-8") as f:
    cases_90_percent = json.load(f)

# Function to find tenor by ID in the 90% dataset
def find_tenor_by_id(case_id, case_data):
    for case in case_data:
        if case.get("id") == case_id:
            return case.get("structured_content", {}).get("tenor", "N/A")
    return "Tenor nicht gefunden"
def find_tatbestand_summary_by_id(case_id, case_data):
    for case in case_data:
        if case.get("id") == case_id:
            return case.get("tatbestand_summary")
    return "Tatbestand_summary nicht gefunden"
# Function to find tenor by ID in the 90% dataset
def find_entscheidungsgruende_by_id(case_id, case_data):
    for case in case_data:
        if case.get("id") == case_id:
            return case.get("structured_content", {}).get("entscheidungsgruende", "N/A")
    return "Gruende nicht gefunden"
# Function to enrich the prompt
async def enrich_prompt(tatbestand, matches, withCritique):
    if withCritique:
        # Prepare input lists for batch processing
        input_texts = [f"Tenor: {result['tenor']}\nTatbestand: {result['tatbestand']}\n Entscheidungsgründe: {result['entscheidungsgruende']}" for result in matches]
        tatbestand_list = [tatbestand] * len(matches)  # Duplicate `tatbestand` for each match

        # Call the batch function and ensure a valid list is returned
        summarized_tatbestands = await filter_and_summarize_batch(input_texts, tatbestand_list)

        if summarized_tatbestands is not None:
            # Update matches with the summarized tatbestands
            for result, summary in zip(matches, summarized_tatbestands):
                result['tatbestand'] = summary
        else:
            logging.error("Batch summarization returned None.")
    #Entscheidungsgruende: {match['entscheidungsgruende']}\n
    enriched_prompt = f"""
    Der folgende Tatbestand wurde gefunden:
    {tatbestand}
    
    Hier sind {len(matches)} Urteile und deren extrahierten Tenors, die etwas mit dem Fall zu tun haben könnten:
    """
    if withCritique:
        for i, match in enumerate(matches, start=1):
            enriched_prompt += f"\n--------\nÜbereinstimmung {i}:\nGerichtsbarkeit: {match['jurisdiction']}\n {match['level_of_appeal']}\nZusammenfassung: {match['tatbestand']}\n\n--------\n"
    else:       
        for i, match in enumerate(matches, start=1):
            enriched_prompt += f"\n--------\nÜbereinstimmung {i}:\nGerichtsbarkeit: {match['jurisdiction']}\n {match['level_of_appeal']}\nTatbestand: {match['tatbestand']}\nTenor: {match['tenor']}\n{match['entscheidungsgruende']}\n--------\n"
    
    enriched_prompt += "\nBitte geben Sie den wahrscheinlichsten Tenor im JSON-Format gemäß der Schema-Angaben zurück."
    enriched_prompt += """{
        "winner": "Beklagter/Kläger",
        "confidence_notes": "string"
    }"""
    return enriched_prompt

# Function to predict tenor with structured output
async def predict_tenor(tatbestand, cases_data, withCritique, withFaiss, query_jurisdiction, case_id):
    # Hole die Top-5-ähnlichsten Dokumente
    k=5
    if withFaiss:
        # Retrieve the top k documents via FAISS
        # Retrieve more than k docs so that after filtering by jurisdiction we still have enough candidates.
        candidate_docs = vectorstore.similarity_search(tatbestand, k=k * 2)
        if query_jurisdiction:
            # Only select docs whose metadata exists and whose 'jurisdiction' (if present) matches the query.
            retrieved_docs = [
                doc for doc in candidate_docs
                if doc.metadata 
                and doc.metadata.get("jurisdiction", "")  # returns "" if key is missing
                and doc.metadata.get("jurisdiction", "").lower() == query_jurisdiction.lower()
            ]
        else:
            retrieved_docs = candidate_docs
        # Take only the top k (if available)
        retrieved_docs = retrieved_docs[:k]
           
    else:
        retrieved_docs = get_top_bm25_matches_for_section(
            tatbestand,
            bm25_tatbestand_index,
            bm25_tatbestand_metadata,
            "tatbestand",
            k=5
        )
    matches = []
    for doc in retrieved_docs:
        # If using FAISS, doc is an object with attributes.
        if withFaiss:
            content = doc.page_content.strip()
            doc_id = doc.metadata.get("id", "Unbekannte ID")
            jurisdiction = doc.metadata.get("jurisdiction", "")
            level_of_appeal = doc.metadata.get("level_of_appeal", "")
        else:
            # If using BM25, doc is a dictionary.
            content = doc.get("tatbestand", "").strip()
            doc_id = doc.get("id", "Unbekannte ID")
            # BM25 metadata currently does not contain extra fields.
            jurisdiction = ""
            level_of_appeal = ""
        
        if not content:
            print(f"Warnung: Leeres Dokument für ID: {doc_id}")
            continue  # Skip this document
        
        tatbestand_summary = find_tatbestand_summary_by_id(doc_id, cases_data)
        tenor = find_tenor_by_id(doc_id, cases_data)
        entscheidungsgruende = find_entscheidungsgruende_by_id(doc_id, cases_data)
        matches.append({
            "tatbestand": tatbestand_summary,
            "tenor": tenor,
            "entscheidungsgruende": entscheidungsgruende,
            "jurisdiction": jurisdiction,
            "level_of_appeal": level_of_appeal
        })


    # Anreichern des Prompts
    # Definiere den System-Prompt
    system_prompt = "Sie sind ein neutraler juristischer Assistent. Analysieren Sie den folgenden Tatbestand und die hinzugefügten Informationen sagen Sie den Tenor vorher. Denken sie gründlich über das Ergebnis nach und bewerten sie neutrale ohne Kläger oder Beklagten zu bevorzugen. Das Ergebnis muss strikt dem vorgegebenen Schema entsprechen."
    tatbestand_summary = find_tatbestand_summary_by_id(case_id, cases_data)
    # Kombiniere System- und User-Prompt
    #Hier in Tatbestand zurückändern falls es wieder normal gelaufen werden soll
    user_prompt = await enrich_prompt(tatbestand_summary, matches, withCritique)
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"


    try:
        response = structured_llm.invoke(combined_prompt)
        return response.dict()  # Konvertiere Pydantic-Objekt in ein Dictionary
    except Exception as e:
        print(f"Fehler bei der Vorhersage: {e}")
        return None

async def main():
    input_file = "enriched_cases_with_summaries.json"
    output_file = "naive_rag_case_predictions_k5_juristdiction-filter_gruende_withCritique_summary.json"
    withCritique = True
    withFaiss = True
    withJuristdictionFilter = True
    # Load the Tatbestand cases
    with open(input_file, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    predictions = []

    for case in tqdm(cases[:100], desc="Processing Cases"):  # Process first 1 case for demonstration
        tatbestand = case.get("structured_content", {}).get("tatbestand", "")
        case_id = case.get("id")
        if withJuristdictionFilter:
            query_jurisdiction = case.get("court", {}).get("jurisdiction", "")
        else: 
            query_jurisdiction = None
        if tatbestand:
            prediction = await predict_tenor(tatbestand, cases_90_percent, withCritique, withFaiss, query_jurisdiction, case_id)

            if prediction:
                # Normalize and format prediction
                prediction["winner"] = normalize_winner_name(prediction.get("winner", ""))
                predictions.append({
                    "case_id": case.get("id"),
                    "tatbestand": tatbestand,
                    "predicted_tenor": prediction,
                })

    # Save predictions to output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)

    print(f"Predictions saved to {output_file}.")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())

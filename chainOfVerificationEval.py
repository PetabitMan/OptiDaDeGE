from pydantic import BaseModel
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
import os
import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm
from UtilityFunctions.evaluateGenerativeMetrics import evaluate_generative_metrics
import asyncio

# Für die Similarity Search der Entscheidungsgründe
from langchain_community.vectorstores import FAISS
# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
AUFGABEN = [
    "Identifizieren Sie alle relevanten Rechtsnormen und Paragraphen, die auf diesen Fall anwendbar sind.",
    "Erläutern Sie, wie diese Normen auf den vorliegenden Fall angewendet werden können.",
    "Liefern Sie eine präzise und logische Argumentation, warum der Beklagte auf Basis von Rechtsnormen gewinnen könnte.",
    "Liefern Sie eine präzise und logische Argumentation, warum der Kläger auf Basis von Rechtsnormen gewinnen könnte.",
    "Verweisen Sie explizit auf Paragraphen und fügen Sie Beispiele aus ähnlichen Fällen hinzu.",
    "Geben Sie eine abschließende Bewertung ab."
]
# --- Embeddings und Vektorstore laden ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
# Hier wird ein FAISS-Index für Entscheidungsgründe erwartet, der neben dem Text (Entscheidungsgründe)
# auch die Felder "tenor" und "tatbestand" in den Metadaten enthält.
decision_reason_vectorstore = FAISS.load_local("faiss_legal_index_entscheidungsgruende", embeddings, allow_dangerous_deserialization=True)

def count_tokens(text, model="text-embedding-3-small"):
    """
    Count the number of tokens in a given text using a specified model.
    """
    tokenizer = tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(text))

# Function to find tenor by ID in the 90% dataset
def find_tenor_by_id(case_id, case_data):
    for case in case_data:
        if case.get("id") == case_id:
            return case.get("structured_content", {}).get("tenor", "N/A")
    return "Tenor nicht gefunden"
# Function to find tenor by ID in the 90% dataset
def find_tatbestand_by_id(case_id, case_data):
    for case in case_data:
        if case.get("id") == case_id:
            return case.get("structured_content", {}).get("tatbestand", "N/A")
    return "Gruende nicht gefunden"
# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# Modelle für strukturierte Ausgaben

class Questions(BaseModel):
    ids: list[int]
    context: list[str]
    content: list[str]

class VerificationResult(BaseModel):
    answer: str
    legitimacy: bool

class CorrectedReasons(BaseModel):
    corrected_decision_reasons: str
    confidence_notes: str
# Load the additional case data
with open("training_cases.json", "r", encoding="utf-8") as f:
    cases_90_percent = json.load(f)

# Initialize LLM mit strukturierter Ausgabe
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
structured_llm_questions = llm.with_structured_output(Questions)
structured_llm_verification = llm.with_structured_output(VerificationResult)

def get_similar_decision_reasons(query,query_jurisdiction, k=2):
    """
    Führt einen Similarity‑Search im Vektorstore der Entscheidungsgründe aus.
    Erwartet wird, dass in den Metadaten auch 'tenor' und 'tatbestand' gespeichert sind.
    """

    candidate_docs = decision_reason_vectorstore.similarity_search(query, k=k * 3)
    
            # Only select docs whose metadata exists and whose 'jurisdiction' (if present) matches the query.
    retrieved_docs = [
        doc for doc in candidate_docs
        if doc.metadata 
        and doc.metadata.get("jurisdiction", "")  # returns "" if key is missing
        and doc.metadata.get("jurisdiction", "").lower() == query_jurisdiction.lower()
    ]
    retrieved_docs = retrieved_docs[:k]

    results = []
    for doc in retrieved_docs:
        content = doc.page_content.strip()
        case_id = doc.metadata.get("id", "Nicht gefunden") if doc.metadata else "Nicht gefunden"
        tenor = find_tenor_by_id(case_id, cases_90_percent)
        tatbestand = find_tatbestand_by_id(case_id, cases_90_percent)
        results.append({
            "entscheidungsgruende": content,
            "tenor": tenor,
            "tatbestand": tatbestand
        })
    return results
# def get_similar_tatbestand_reasons(query, k=2):
#     """
#     Führt einen Similarity‑Search im Vektorstore der Entscheidungsgründe aus.
#     Erwartet wird, dass in den Metadaten auch 'tenor' und 'tatbestand' gespeichert sind.
#     """
#     candidate_docs = tatbestand_vectorstore.similarity_search(tatbestand, k=k * 3)
#             # Only select docs whose metadata exists and whose 'jurisdiction' (if present) matches the query.
#     retrieved_docs = [
#         doc for doc in candidate_docs
#         if doc.metadata 
#         and doc.metadata.get("jurisdiction", "")  # returns "" if key is missing
#         and doc.metadata.get("jurisdiction", "").lower() == query_jurisdiction.lower()
#     ]
#     retrieved_docs = retrieved_docs[:k]

#     results = []
#     for doc in retrieved_docs:
#         content = doc.page_content.strip()
#         case_id = doc.metadata.get("id", "Nicht gefunden") if doc.metadata else "Nicht gefunden"
#         tenor = find_tenor_by_id(case_id, cases_90_percent)
#         entscheidungsgruende = find_entscheidungsgruende_by_id(case_id, cases_90_percent)
#         results.append({
#             "entscheidungsgruende": entscheidungsgruende,
#             "tenor": tenor,
#             "tatbestand": content
#         })
#     return results
def process_decision_reasons_in_parts(tatbestand: str, hypothetical_reasons: str, query_jurisdiction: str, role: str) -> str:
    """
    Teilt die hypothetischen Entscheidungsgründe in sinnvolle Teile,
    verarbeitet jeden Teil einzeln mittels chain_of_verification und fügt die Ergebnisse zusammen.
    """
    # Text in Teile aufteilen
    parts = split_text(hypothetical_reasons, max_words=500)
    transformed_parts = []
    for idx, part in enumerate(parts, start=1):
        # Du kannst hier optional auch den Chunk-Index in den Prompt integrieren, falls relevant.
        transformed_part = chain_of_verification(tatbestand, part, query_jurisdiction, role=role, iteration=idx)
        transformed_parts.append(transformed_part)
    
    # Alle transformierten Teile zusammenfügen
    full_transformed_text = "\n\n".join(transformed_parts)
    return full_transformed_text

def split_text(text: str, max_words: int = 500) -> list[str]:
    """
    Teilt einen Text in Chunks auf, die höchstens max_words Wörter enthalten.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
async def chain_of_verification_async(tatbestand: str, hypothetical_reasons: str, query_jurisdiction: str, role: str, iteration: int = 1) -> str:
    # Falls chain_of_verification synchron ist, führe es in einem Thread aus.
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,  # None verwendet den Standard ThreadPoolExecutor
        chain_of_verification,  # synchroner Funktionsaufruf
        tatbestand, hypothetical_reasons, role, iteration
    )

async def process_decision_reasons_in_parts(tatbestand: str, hypothetical_reasons: str, query_jurisdiction: str, role: str) -> str:
    # Aufteilen des langen Textes in sinnvolle Teile (hier z. B. 300 Wörter pro Chunk)
    parts = split_text_advanced(text=hypothetical_reasons,method="direct", aufgaben=AUFGABEN ,multi_query_chunk_size=700)
 

    # Erstelle für jeden Chunk einen Task, der chain_of_verification asynchron ausführt.
    tasks = [
        chain_of_verification_async(tatbestand, part, query_jurisdiction=query_jurisdiction, role=role, iteration=idx)
        for idx, part in enumerate(parts, start=1)
    ]
    # Warte, bis alle Tasks abgeschlossen sind.
    transformed_parts = await asyncio.gather(*tasks)
    # Füge alle Ergebnisse zusammen
    full_transformed_text = "\n\n".join(transformed_parts)
    return full_transformed_text

def split_text_advanced(text: str, method: str, aufgaben: list[str] = None, multi_query_chunk_size: int = 700) -> list[str]:
    """
    Teilt den Text abhängig von der gewählten Methode:
      - "direct": Kein Split, der gesamte Text wird als einzelner Chunk zurückgegeben.
      - "multi-query": Aufteilung in Chunks à multi_query_chunk_size Wörter.
      - "least-to-most": Splitten an den Stellen, an denen Aufgaben aus der Liste `aufgaben` im Text gefunden werden.
    """
    if method == "direct":
        # Kein Split – der gesamte Text wird in einer Liste zurückgegeben
        return [text]

    elif method == "multi-query":
        # Aufteilung in Chunks mit jeweils maximal multi_query_chunk_size Wörtern
        words = text.split()
        chunks = []
        current_chunk = []
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= multi_query_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    elif method == "least-to-most":
        # Wenn eine Liste von Aufgaben übergeben wurde, versuchen wir den Text an diesen Stellen zu splitten.
        # Dabei gehen wir den Text sequenziell durch und suchen nach jedem Aufgabentext.
        if aufgaben:
            parts = []
            remaining_text = text
            for aufgabe in aufgaben:
                # Suchen, ob die Aufgabe im verbleibenden Text vorkommt
                idx = remaining_text.find(aufgabe)
                if idx != -1:
                    part = remaining_text[:idx].strip()
                    if part:
                        parts.append(part)
                    # Den gefundenen Aufgabentext sowie alles davor entfernen,
                    # sodass im nächsten Durchlauf der restliche Text betrachtet wird.
                    remaining_text = remaining_text[idx:]
            if remaining_text.strip():
                parts.append(remaining_text.strip())
            # Falls nichts gefunden wurde, gib den Originaltext zurück
            return parts if parts else [text]
        else:
            return [text]
    else:
        raise ValueError(f"Invalid method: {method}")

def chain_of_verification(tatbestand: str, hypothetical_reasons: str, query_jurisdiction: str, role: str, iteration: int = 1) -> str:
    """
    Chain-of-Verification-Prozess mit strukturierter Ausgabe und angepasstem Schritt 3.
    """
    query = hypothetical_reasons
    similar_matches = get_similar_decision_reasons(query, query_jurisdiction, k=2)
    similar_str = ""
    for match in similar_matches:
        similar_str += (
            f"---------"
            f"\nEntscheidungsgründe: {match['entscheidungsgruende']}\n"
            f"Tenor: {match['tenor']}\n"
            f"Tatbestand: {match['tatbestand']}\n"
            f"---------"
        )
    # Schritt 1: Strukturierte Fragen generieren
    question_prompt = f"""
        Sie agieren als ein neutraler, hochspezialisierter juristischer Assistent mit fundiertem Fachwissen. Ihre Aufgabe besteht darin, den untenstehenden Tatbestand sowie die hypothetischen Entscheidungsgründe eingehend zu analysieren und daraus gezielte, strukturierte Fragen abzuleiten, die:
        - Relevante Rechtsnormen und Paragraphen identifizieren,
        - Juristische Argumentationslinien kritisch hinterfragen und
        - Bezüge zu ähnlichen, vergangenen Gerichtsurteilen herstellen.

        Tatbestand:
        {tatbestand}

        ------
        Hypothetische Entscheidungsgründe:
        {hypothetical_reasons}

        ------
        Ähnliche Entscheidungsgründe aus vergangenen Gerichtsurteilen (inklusive Tenor und Tatbestand):
        {similar_str}

        Bitte geben Sie die resultierenden Fragen in folgendem JSON-Format zurück:
        {{
    
        "ids": [1, 2, ...],
        "context": ["Relevanter context 1", "Relevanter context 2, ..."],
        "content": ["Frage 1", "Frage 2, ..."]
        }}
        """

    questions_response = structured_llm_questions.invoke(question_prompt, temperature=0)
    
    questions = questions_response.content
    contexts = questions_response.context

    # Schritt 2: Fragen verifizieren
    verification_results = []
    for question, context in zip(questions, contexts):

        
        verification_prompt = f"""
            Sie sind ein neutraler juristischer Gutachter. Bitte beantworten Sie die folgende Frage unter Berücksichtigung des angegebenen Kontexts und bewerten Sie gleichzeitig, ob die in der Fragestellung enthaltenen juristischen Argumente fundiert und stichhaltig sind.

            Frage:
            {question}

            Kontext:
            {context}

            Bitte liefern Sie Ihre Antwort im folgenden JSON-Format:
            {{
                "answer": "<Ihre präzise und juristisch begründete Antwort>",
                "legitimacy": <true oder false, je nachdem, ob die Argumentation als juristisch legitim bewertet wird>
            }}

            Hinweis: Nutzen Sie klare juristische Terminologie und begründen Sie kurz, warum Sie die Argumentation als legitim oder nicht legitim einstufen.
            """

        verification_response = structured_llm_verification.invoke(verification_prompt, temperature=0)

        verification_result = verification_response.dict()
        # Add original question and context to verification result
        verification_result.update({"original_question": question})
        verification_results.append(verification_result)

    # Schritt 3: Korrigierte Entscheidungsgründe generieren
    
    verification_results_str = json.dumps(verification_results, indent=4)
    correction_prompt = f"""
        Sie agieren als ein neutraler, hochqualifizierter juristischer Assistent. Im Folgenden finden Sie:

        1. Den Tatbestand des Falls,
        2. Die vorliegenden hypothetischen Entscheidungsgründe,
        3. Die Verifikationsergebnisse zu einzelnen juristischen Fragestellungen (inklusive der Bewertungen und ergänzenden Erläuterungen).

        Ihre Aufgabe ist es, die hypothetischen Entscheidungsgründe auf Basis der Verifikationsergebnisse umfassend zu überarbeiten. Bitte beachten Sie dabei folgende Vorgaben:
        - **Integration:** Integrieren Sie die einzelnen, juristisch fundierten Antworten aus den Verifikationsergebnissen in Ihre Überarbeitung.
        - **Korrektur:** Identifizieren und korrigieren Sie alle Passagen in den Entscheidungsgründen, die nicht den tatsächlichen rechtlichen Gegebenheiten oder einer stichhaltigen Argumentation entsprechen.
        - **Erweiterung und Konkretisierung:** Sorgen Sie dafür, dass sämtliche juristischen Argumentationsstränge ausführlich erläutert werden. Fügen Sie – sofern aus den Verifikationsergebnissen hervorgeht – relevante Rechtsnormen, Präzedenzfälle und Beispiele hinzu.
        - **Transparenz:** Ihre Überarbeitung soll als fortlaufender, zusammenhängender Fließtext erfolgen, der für Dritte nachvollziehbar und juristisch stringent ist.

        Bitte verwenden Sie folgenden Aufbau:

        ---
        **Tatbestand:**
        {tatbestand}

        **Vorliegende hypothetische Entscheidungsgründe:**
        {hypothetical_reasons}

        **Verifikationsergebnisse (als JSON):**
        {verification_results_str}
        ---

        Geben Sie abschließend den überarbeiteten Fließtext der Entscheidungsgründe zurück.
        """


        
    correction_response = llm.invoke(correction_prompt)
    if hasattr(correction_response, 'content'):
        return correction_response.content.strip()
    else:
        return correction_response.strip()
async def evaluate_transformed_hypothetical(hypothetical_file: str, output_file: str):
    print("loading hypothetical_cases")
    with open(hypothetical_file, 'r', encoding='utf-8') as f:
        hypothetical_data = json.load(f)
    chosen_method = "multi-query"
    results = []
    role = "Kläger"
    for entry in tqdm(hypothetical_data, desc="Evaluating Transformed Hypotheticals"):
        case_id = entry.get('case_id')
        method = entry.get('method')
        if method != chosen_method:
            continue
        original_metrics = entry.get('metrics')
        tatbestand = entry.get('tatbestand', '').strip()
        generated_entscheidungsgruende = entry.get('generated_entscheidungsgruende', '').strip()
        actual_entscheidungsgruende = entry.get('actual_entscheidungsgruende', '').strip()
        if not tatbestand or not generated_entscheidungsgruende:
            continue
        query_jurisdiction = entry.get("court", {}).get("jurisdiction", "")

        # Verwende die asynchrone, parallelisierte Version:
        transformed_reasons = await process_decision_reasons_in_parts(tatbestand, generated_entscheidungsgruende, query_jurisdiction, role=role)

        metrics = evaluate_generative_metrics(actual_entscheidungsgruende, transformed_reasons)

        results.append({
            "case_id": case_id,
            "role": role,
            "method": method,
            "tatbestand": tatbestand,
            "original_hypothetical": generated_entscheidungsgruende,
            "transformed_hypothetical": transformed_reasons,
            "metrics_transformed_actual": metrics,
            "metrics_generated_actual": original_metrics,
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Evaluation completed and saved to {output_file}")



if __name__ == "__main__":
    chosen_method_name="multi-query"
    hypothetical_file = "hyde_comparison_results_batched_unbiased.json"
    output_file = f"chain-of-verification_evaluation_results_{chosen_method_name}.json"
    
    # Starte das asynchrone Evaluierungs-Skript
    asyncio.run(evaluate_transformed_hypothetical(hypothetical_file, output_file))

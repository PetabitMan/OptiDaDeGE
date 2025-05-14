from pydantic import BaseModel
import json
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
import os
import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm
from UtilityFunctions.evaluateGenerativeMetrics import evaluate_generative_metrics
import asyncio

def count_tokens(text, model="text-embedding-3-small"):
    """
    Count the number of tokens in a given text using a specified model.
    """
    tokenizer = tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(text))
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

# Initialize LLM mit strukturierter Ausgabe
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
structured_llm_questions = llm.with_structured_output(Questions)
structured_llm_verification = llm.with_structured_output(VerificationResult)
BATCH_SIZE = 10  # Number of prompts per batch

AUFGABEN = [
    "Identifizieren Sie relevante Rechtsnormen und Paragraphen, die auf diesen Fall anwendbar sind.",
    "Erläutern Sie, wie diese Normen auf den vorliegenden Fall angewendet werden können.",
    "Liefern Sie eine präzise und logische Argumentation, warum der Kläger oder der Beklagte auf Basis dieser Rechtsnormen gewinnen könnte.",
    "Verweisen Sie explizit auf Paragraphen und fügen Sie Beispiele aus ähnlichen Fällen hinzu.",
    "Geben Sie eine abschließende Bewertung ab."
]


async def generate_entscheidungsgruende_batched(tatbestand_list, method="direct", use_verification=False):
    results = []

    if method == "direct":
        prompts = [
            f"""
            Sie sind ein hochspezialisierter juristischer Assistent und haben die Aufgabe, Entscheidungsgründe für den folgenden Fall zu formulieren. 

            **Tatbestand:**
            {tatbestand}

            **Aufgabe:**
            1. {AUFGABEN[0]}
            2. {AUFGABEN[1]}
            3. {AUFGABEN[2]}
            4. {AUFGABEN[3]}
            5. {AUFGABEN[4]}
            """
            for tatbestand in tatbestand_list
        ]
        responses = await llm.agenerate(prompts)
        sub_results = [generation[0].text for generation in responses.generations]

        # Apply Chain of Verification if enabled
        if use_verification:
            for idx, sub_result in enumerate(sub_results):
                sub_results[idx] = chain_of_verification(tatbestand_list[idx], sub_result, "direct")
        
        results.extend(sub_results)
    elif method == "multi-query":
        for tatbestand in tatbestand_list:
            prompts = [f"{aufgabe}\n\nTatbestand:\n{tatbestand}" for aufgabe in AUFGABEN]
            responses = await llm.agenerate(prompts)
            sub_results = [generation[0].text for generation in responses.generations]

            if use_verification:
                for idx, sub_result in enumerate(sub_results):
                    sub_results[idx] = chain_of_verification(tatbestand, sub_result, "multi-query")
            
            results.append("\n\n".join(sub_results))

    elif method == "least-to-most":
        contexts = [f"Tatbestand:\n{tatbestand}\n\n" for tatbestand in tatbestand_list]

        for aufgabe in AUFGABEN:
            prompts = [f"{context}{aufgabe}" for context in contexts]
            responses = await llm.agenerate(prompts)

            for idx, generation in enumerate(responses.generations):
                new_content = generation[0].text

                if use_verification:
                    new_content = chain_of_verification(contexts[idx], new_content, "least-to-most")

                contexts[idx] += f"\n\n{aufgabe}:\n{new_content}"

        results.extend(contexts)

    else:
        raise ValueError(f"Invalid method: {method}")

    return results


def chain_of_verification(tatbestand: str, hypothetical_reasons: str, role: str, iteration: int = 1) -> str:
    """
    Chain-of-Verification-Prozess mit strukturierter Ausgabe und angepasstem Schritt 3.
    """
    # Schritt 1: Strukturierte Fragen generieren
    question_prompt = f"""
    Sie sind ein spezialisierter juristischer Assistent für den {role}. Analysieren Sie den folgenden Tatbestand und die hypothetischen Entscheidungsgründe und erstellen Sie strukturierte Fragen, die relevante Aspekte beleuchten und geben sie dazu den Kontext der Frage mit.
    Beachten sie dabei den Beibringungsgrundsatz. Stellen sie also nur Fragen zur rechtlichen Auslegung des Tatbestandes und keine Fragen zum Tatbestand selbst. 

    Tatbestand:
    {tatbestand}

    Hypothetische Entscheidungsgründe:
    {hypothetical_reasons}
    Ähnliche Entscheidungsgründe aus vergangen Gerichtsurteilen:
    {#hier matches einfügen}
    Geben Sie die Fragen im JSON-Format zurück:
 
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
    # Ensure questions and contexts are correctly paired
    for question, context in zip(questions, contexts):
        print(question)
        verification_prompt = f"""
        Frage: {question}
        Context: {context}
        Beantworten Sie diese Frage und bewerten Sie die Legitimität der Argumente.
        Geben Sie die Antwort im JSON-Format zurück:
        {{
            "answer": "Antwort",
            "legitimacy": true/false
        }}
        """
        verification_response = structured_llm_verification.invoke(verification_prompt, temperature=0)

        verification_result = verification_response.dict()
        # Add original question and context to verification result
        verification_result.update({"original_question": question})
        verification_results.append(verification_result)
    # Schritt 3: Korrigierte Entscheidungsgründe generieren
    
    verification_results_str = json.dumps(verification_results, indent=4)
    correction_prompt = f"""
        Du bist ein juristischer Assistent. Unten stehen:
        1. Tatbestand,
        2. Hypothetische Entscheidungsgründe,
        3. Verifikationsergebnisse pro Frage.

        Basierend auf den Verifikationsergebnissen sollst du die hypothetischen Entscheidungsgründe detailliert überarbeiten:
            1. Integriere die einzelnen Antworten aus den Verifikationsergebnissen in die überarbeiteten Entscheidungsgründe.
            2. Ändere alles in den hypothetischen Entscheidungsgründen, was sich als falsch erwiesen hat
            3. Achte darauf, sämtliche juristischen Argumentationsstränge ausführlich auszuleuchten und diese gegebenenfalls zu erweitern oder zu konkretisieren, sofern es aus den Fragen/Antworten hervorgeht.

        Gib das Ergebnis als Fließtext zurück, der die hypothethischen Entscheidungsgründe zurückgibt und dabei alle oben genannten Anforderungen erfüllt.

        TATBESTAND:
        {tatbestand}

        HYPOTHETISCHE ENTSCHEIDUNGSGRÜNDE:
        {hypothetical_reasons}

        VERIFIKATIONSERGEBNISSE:
        {verification_results_str}
    """

        
    correction_response = llm.invoke(correction_prompt)
    print(correction_response)
    if hasattr(correction_response, 'content'):
        return correction_response.content.strip()
    else:
        return correction_response.strip()

async def main():
    with open("enriched_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    methods = ["multi-query", "least-to-most", "direct"]
    results = []

    for method in methods:
        for i in tqdm(range(0, len(cases[:10]), BATCH_SIZE), desc=f"Processing Batches for {method}"):
            batch = cases[i : i + BATCH_SIZE]
            tatbestand_list = [
                case.get("structured_content", {}).get("tatbestand", "").strip()
                for case in batch
                if case.get("structured_content", {}).get("tatbestand", "").strip()
            ]
            actual_entscheidungsgruende_list = [
                case.get("structured_content", {}).get("entscheidungsgruende", "").strip()
                for case in batch
                if case.get("structured_content", {}).get("entscheidungsgruende", "").strip()
            ]

            if not tatbestand_list:
                continue

            # Generate Entscheidungsgründe for the batch
            use_verification = True  # Enable or disable Chain of Verification here
            generated_list = await generate_entscheidungsgruende_batched(tatbestand_list, method=method, use_verification=use_verification)

            # Compute metrics and save results
            for case, generated, actual in zip(batch, generated_list, actual_entscheidungsgruende_list):
                metrics = evaluate_generative_metrics(actual, generated)
                results.append({
                    "case_id": case.get("id"),
                    "tatbestand": case.get("structured_content", {}).get("tatbestand", "").strip(),
                    "actual_entscheidungsgruende": actual,
                    "method": f"{method} ({'with verification' if use_verification else 'without verification'})",
                    "generated_entscheidungsgruende": generated,
                    "metrics": metrics,
                })

    # Save results
    output_file = f"hyde_comparison_results_batched_{'with_verification' if use_verification else 'without_verification'}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Batch processing completed and saved to {output_file}.")


if __name__ == "__main__":
    asyncio.run(main())
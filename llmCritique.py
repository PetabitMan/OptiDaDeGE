import os
import logging
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import List
import asyncio
import logging
# Load OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, verbose=False)
async def generate_with_retries(llm, prompts, max_retries=5, wait_time=60):
    """Call llm.agenerate(prompts) with retries upon encountering an error."""
    for attempt in range(max_retries):
        try:
            responses = await llm.agenerate(prompts)
            return responses  # Return if successful
        except Exception as e:
            print(
                f"Error encountered: {e}\n"
                f"Retrying in {wait_time} seconds (attempt {attempt + 1}/{max_retries})."
            )
            await asyncio.sleep(wait_time)

    # Raise an exception if all retries fail
    raise Exception(
        f"Failed to call llm.agenerate after {max_retries} retries due to repeated errors."
    )


if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment.")

# Configure logging to suppress verbose output
logging.basicConfig(level=logging.WARNING)
# Define the prompt template
SUMMARY_PROMPT = """
    Sie sind ein neutraler juristischer Assistent. Ihre Aufgabe ist es, relevante Informationen aus den folgenden verwandten Gerichtsurteilen und Entscheidungsgründen zu extrahieren
    und diese auf den Tatbestand zu beziehen, so dass ein Richter damit eine fundierte Entscheidung treffen kann. Denken sie gründlich über ihr Ergebnis nach und bevorzugen sie wede Kläger noch Beklagten:
    Bevorzugen sie weder Kläger noch den Beklagten.
    Tatbestand:
    {tatbestand}

    Eingabetext:
    {input_text}

    Bitte geben Sie relevante Informationen zurück, die mit dem Tatbestand zusammenhängen,
    und bereiten Sie diese so auf, dass ein Richter eine gut fundierte Entscheidung treffen kann.
    """
def filter_and_summarize(input_text: str, tatbestand: str) -> str:
    """
    Processes input text to extract and summarize relevant information based on the given Tatbestand.

    Parameters:
        input_text (str): The input string containing all information.
        tatbestand (str): The Tatbestand (case facts) guiding the relevance extraction.

    Returns:
        str: Summarized relevant information as a single cohesive output.
    """
    # Define the prompt
    summary_prompt = SUMMARY_PROMPT.format(tatbestand=tatbestand, input_text=input_text)
    try:
        # Generate the summary
        response = llm.invoke(
            summary_prompt.format(
                tatbestand=tatbestand,
                input_text=input_text
            )
        )
        # Extract summary content
        summary = response.content if hasattr(response, "content") else str(response)
        return summary
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        return ""


async def filter_and_summarize_batch(input_texts: List[str], tatbestands: List[str]) -> List[str]:
    """
    Processes multiple input texts to extract and summarize relevant information based on the given Tatbestands.

    Parameters:
        input_texts (List[str]): A list of input strings containing all information.
        tatbestands (List[str]): A list of Tatbestand strings guiding the relevance extraction.

    Returns:
        List[str]: A list of summarized relevant information for each input.
    """
    summaries=[]
    if len(input_texts) != len(tatbestands):
        raise ValueError("Input texts and Tatbestands must have the same length.")

    prompts = [
        SUMMARY_PROMPT.format(tatbestand=tatbestand, input_text=input_text)
        for tatbestand, input_text in zip(tatbestands, input_texts)
    ]

    try:
        responses = await generate_with_retries(llm, prompts)
        summaries.extend(generation[0].text for generation in responses.generations)
        return summaries
    except Exception as e:
        logging.error(f"Error during batch summarization: {e}")
        return ["Error during summarization"] * len(input_texts)
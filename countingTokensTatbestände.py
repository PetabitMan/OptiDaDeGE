import json
from dotenv import load_dotenv
import tiktoken
from tqdm import tqdm

print("Loading Cases...")
with open("training_cases.json", "r", encoding="utf-8") as f:
    cases_90_percent = json.load(f)
def count_tokens(text, model="text-embedding-3-small"):
    """
    Count the number of tokens in a given text using a specified model.
    """
    tokenizer = tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(text))

sumTotal=0 
for case in tqdm(cases_90_percent, desc="Processing Documents"):
    tatbestand = case.get("structured_content", {}).get("tatbestand", "")
    sumTotal += count_tokens(tatbestand, model="text-embedding-3-small")

print(sumTotal)
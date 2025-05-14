import os
import json
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import HumanMessage

# 1) Environment & LLM
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.3)

# 2) Prompt-Template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Du bist ein hilfreicher juristischer Assistent. Fasse den folgenden Tatbestand präzise und knapp zusammen:"
    ),
    HumanMessagePromptTemplate.from_template("{tatbestand}")
])

def summarize_tatbestand(tatbestand: str) -> str:
    messages = prompt.format_prompt(tatbestand=tatbestand).to_messages()
    llm_result = llm.generate([messages])
    return llm_result.generations[0][0].text.strip()

def main():
    with open("enriched_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    for case in tqdm(cases[:100], desc="Summarizing Tatbestand"):
        tb = case.get("structured_content", {}).get("tatbestand", "")
        case["tatbestand_summary"] = summarize_tatbestand(tb) if tb else ""

    with open("enriched_cases_with_summaries.json", "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print("Fertig! -> enriched_cases_with_summaries.json")

if __name__ == "__main__":
    main()

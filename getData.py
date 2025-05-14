import json
import re
from datetime import date
# Clean text function
from tqdm import tqdm
import html
# Custom function to handle non-serializable types like dates
def json_serializer(obj):
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")



# Clean text function
def clean_text(text):
    """
    Cleans legal case text by decoding HTML entities, removing HTML tags, 
    extra spaces, and normalizing the text.
    """
    # Decode HTML entities
    text = re.sub(r'\n', ' --NEWLINE-- ', text)
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    # Remove excess spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
def remove_new_line(text):
    text = re.sub(r'--NEWLINE--', ' ', text)
    return text
def extract_sections(text):
    """
    Extracts the relevant sections from the legal text, with enhanced patterns
    to handle unusual formats.
    """
    cleaned_text = clean_text(text)
    
    # Enhanced patterns to handle unusual formats
    tenor_pattern = r"(Tenor\s.*?)(Tatbestand:|T a t b e s t a n d:|Gründe:|E n t s c h e i d u n g s g r ü n d e:|$)"
    # Allow for spaced-out letters in "Tatbestand"
    tatbestand_pattern = r"(T\s?a\s?t\s?b\s?e\s?s\s?t\s?a\s?n\s?d\s?:\s.*?)(E n t s c h e i d u n g s g r ü n d e:|Entscheidungsgründe:|Gründe:|$)"
    #entscheidungsgruende_pattern = r"(g\s?r\s?ü\s?n\s?d\s?e\s?:)\s*(.*)"
    #entscheidungsgruende_pattern = r"g\s?r\s?ü\s?n\s?d\s?e\s?:\s*(.*)"
    entscheidungsgruende_pattern = r"(g\s?r\s?ü\s?n\s?d\s?e\s?:\s*.*)"

    #entscheidungsgruende_pattern = r"(g\s?r\s?ü\s?n\s?d\s?e\s?:)\s*(.*)"


    # Extract 'tenor' section
    tenor_match = re.search(tenor_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
    tenor = tenor_match.group(1).strip() if tenor_match else ""
    tenor = remove_new_line(tenor)
    # Extract 'tatbestand' section (optional with spaced-out letters)
    tatbestand_match = re.search(tatbestand_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
    tatbestand = tatbestand_match.group(1).strip() if tatbestand_match else ""
    tatbestand = remove_new_line(tatbestand)
    # Extract 'entscheidungsgruende' section, handle absence of explicit label
    entscheidungsgruende_match = re.search(entscheidungsgruende_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
    entscheidungsgruende = entscheidungsgruende_match.group(1).strip() if entscheidungsgruende_match else ""
    
    return {
        "tenor": tenor,
        "tatbestand": tatbestand,
        "entscheidungsgruende": entscheidungsgruende
    }

# Extract case references
def extract_case_references(text):
    """
    Extracts legal references from the text using a regex pattern.
    """
    cleaned_text = clean_text(text)
    pattern = r'\d+\s*[A-Za-z]+\s*\d+/\d+'
    matches = re.findall(pattern, cleaned_text)
    case_references = [match.strip() for match in matches]
    return case_references
# Extract legal references
def extract_legal_references(text):
    """
    Extracts legal references from the text using a regex pattern.
    """
    cleaned_text = clean_text(text)
    pattern = r"§\s?\d+(?:\s?[a-zA-Z]*)?(?:\s?Abs\.\s?\d+)?(?:\s?S\.\s?\d+)?(?:\s?Nr\.\s?\d+)?(?:\s?[a-zA-Z]*)?\s?[A-Z]+"
    matches = re.findall(pattern, cleaned_text)
    legal_references = [match.strip() for match in matches]
    return legal_references

# Function to remove 'content' from each case
def remove_content_from_cases(cases):
    for case in cases:
        if 'content' in case:
            del case['content']  # Remove the 'content' field
    return cases

# Load and process the first 100 cases from JSONL file
try:
    # Liste für die modifizierten Fälle
    modified_cases = []
    sufficient_cases = []

    count = 0  # Zähler für die ersten 100 Fälle
    
    # Laden der Fälle aus der JSONL-Datei
    with open('cases.jsonl', 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing data", mininterval=1):
            case = json.loads(line)
            
            # Rohtext (content) extrahieren
            raw_text = case.get('content', '')
            
            # Extrahieren der strukturierten Abschnitte
            sections = extract_sections(raw_text)
            
            # Extrahieren der Rechtsreferenzen
            case_references = extract_case_references(raw_text)
            
            # Modifizieren der Falldaten
            case['structured_content'] = {
                "tenor": sections["tenor"],
                "tatbestand": sections["tatbestand"],
                "entscheidungsgruende": sections["entscheidungsgruende"]
            }
            case['case_references'] = case_references

            # Hinzufügen des modifizierten Falls zur Liste
            modified_cases.append(case)
            # Check if all required fields are non-empty
            if sections["tenor"] and sections["tatbestand"] and sections["entscheidungsgruende"]:
                sufficient_cases.append(case)

         
    modified_cases= remove_content_from_cases(modified_cases)
    # Speichern der modifizierten Fälle als JSON-Datei
    with open('court_cases_structured.json', 'w', encoding='utf-8') as f:
        json.dump(modified_cases, f, ensure_ascii=False, indent=4, default=json_serializer)

    print("Die modifizierten Daten wurden erfolgreich gespeichert.")
   # Save sufficient cases to a separate file
    with open('sufficient_cases.json', 'w', encoding='utf-8') as f:
        json.dump(sufficient_cases, f, ensure_ascii=False, indent=4, default=json_serializer)

    print("Data processed and saved successfully.")
    print(f"Total sufficient cases: {len(sufficient_cases)}")

  
except Exception as e:
    print(f"Fehler beim Verarbeiten der JSONL-Datei: {e}")
#251038
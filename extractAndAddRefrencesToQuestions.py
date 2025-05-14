#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import sys
from tqdm import tqdm

# --- Hier: Liste der einzeln zu verarbeitenden QA-Dateien ---
INPUT_FILES = [
    "questions/stgb_QA.json",
    "questions/zpo_QA.json",
    "questions.json",
    "questions/bgb_eval_qa.json",
    "questions/GerLayQA.json",
]

BOOK_CODES_FILE = "book_codes.json"
OUTPUT_FILE     = "questions_with_refs.json"


def load_book_codes(json_file):
    """Lädt die Liste der Gesetzbuch-Kürzel."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            codes = json.load(f)
            if not isinstance(codes, list):
                raise ValueError("book_codes.json muss eine Liste von Kürzeln sein")
            return codes
    except Exception as e:
        print(f"[Error] Buch-Codes konnten nicht geladen werden: {e}")
        sys.exit(1)


def extract_law_references(text, book_codes, max_length=80):
    """Findet rohe §-Referenzen im Text."""
    codes_pattern = r"|".join(re.escape(code) for code in book_codes)
    regex = re.compile(rf"(§{{1,2}}\s*\d+[^,;\n]*\b(?:{codes_pattern})\b)", re.UNICODE)
    matches = regex.findall(text)
    cleaned = []
    for m in matches:
        m_clean = " ".join(m.split())
        if len(m_clean) <= max_length:
            cleaned.append(m_clean)
    return cleaned


def simplify_references(refs):
    """Teilt Zusammenfassungen wie '§§ 929, 930 BGB' in ['929 BGB','930 BGB']."""
    simple = set()
    for ref in refs:
        m_code = re.search(r"\b([A-Za-z0-9ÖÄÜöäü]+)\s*$", ref)
        code = m_code.group(1) if m_code else ""
        nums = re.findall(r"\d+", ref)
        for n in nums:
            simple.add(f"{n} {code}".strip())
    return sorted(simple)


def normalize_entry(raw, book_codes):
    """Führt jedes Roh-Dict auf das einheitliche Schema zurück."""
    q = raw.get("Question_text") or raw.get("question") or ""
    a = raw.get("Answer_text")   or raw.get("answer") or raw.get("gold_answer")   or ""
    text = f"{q}\n{a}"
    refs = extract_law_references(text, book_codes)
    simple = simplify_references(refs)
    return {
        "question":       q.strip(),
        "answer":         a.strip(),
        "law_references": refs,
        "simple_refs":    simple
    }


def main():
    book_codes = load_book_codes(BOOK_CODES_FILE)
    all_qas = []

    for fname in INPUT_FILES:
        print(f"Verarbeite {fname}…")
        try:
            with open(fname, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Error] Konnte {fname} nicht laden: {e}")
            continue

        if not isinstance(data, list):
            print(f"[Warnung] {fname} enthält kein Array, wird übersprungen.")
            continue

        for raw in tqdm(data, desc=f"Einträge in {fname}"):
            entry = normalize_entry(raw, book_codes)
            all_qas.append(entry)

    # Ausgabe speichern
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_qas, f, ensure_ascii=False, indent=2)

    print(f"\nFertig! Gesamt-Einträge: {len(all_qas)} → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

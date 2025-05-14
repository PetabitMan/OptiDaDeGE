# OptiDaDeGE

**Optimierung der Datenanalyse von deutschen Gerichtsurteilen durch RAG mit LLMs**  
*Legal Case Analysis Toolkit*

Dieses Repository enthält eine Sammlung von Python-Tools zur Analyse, Verarbeitung und Visualisierung juristischer Falldaten (Schwerpunkt: deutsche Gerichtsurteile). Mittels moderner NLP-Methoden, Embeddings und Retrieval-Augmented Generation (RAG) können Sie:

- Tatbestände und Entscheidungsgründe strukturieren  
- Ähnliche Fälle finden und visuell auswerten  
- Vorhersagen zum Tenor und zu Rechtsgrundlagen treffen  
- Statistische Auswertungen und generative Metriken berechnen  

---

## 🚀 Quick Start

1. **API-Key & Environment**  
   - OpenAI-Account und API-Key erstellen  
   - `.env` anlegen und `OPENAI_API_KEY=…` eintragen  

2. **Daten herunterladen**  
   ```bash
   wget https://static.openlegaldata.io/dumps/de/cases.jsonl
   wget https://static.openlegaldata.io/dumps/de/laws.jsonl
   wget https://static.openlegaldata.io/dumps/de/book_codes.jsonl
   ```

3. **Daten aufbereiten & Referenzen extrahieren**  
   ```bash
   python getData.py cases.jsonl
   python extractAndAddRefrences.py
   ```

4. **Embeddings erzeugen & Index erstellen**  
   ```bash
   python generateEmbeddingsTatbestand.py
   python generateEmbeddingsEntscheidungsgründe.py
   python createIndexBM25.py
   ```

5. **(Optional) Multi-Threaded Embeddings**  
   ```bash
   python generateEmbeddingsEntscheidungruende_threads.py
   ```

Nach diesen Schritten steht Ihnen eine vollständige Retrieval-Datenbasis für alle Beispiele und Workflows zur Verfügung.

---

## 📂 Verzeichnisstruktur

### 1. Case Analysis & Prediction
- `casePrediction.py`  
  **Core-Modell**: Vorhersage von Tenor & Kostenübernahme (gpt-4o-mini)  
- `casePredictionSuperSimple.py`  
  **Light-Version**: Minimalistisches Beispiel  
- `casePredicitionReasoning.py`  
  **Mit Begründungen**: Chain-of-Thought für Erklärungen  
- `analysePredictions2.py`  
  **Statistik**: Auswertung & Vergleich der Vorhersageergebnisse  

### 2. Text Processing & Embeddings
- `generateEmbeddingsTatbestand.py`  
  **Tatbestand** → Vektoren  
- `generateEmbeddingsEntscheidungsgründe.py`  
  **Entscheidungsgründe** → Vektoren  
- `generateEmbeddingsEntscheidungruende_threads.py`  
  **Multi-Threaded** für höhere Performance  
- `countingChunks.py`, `countingTokensTatbestände.py`  
  **Analyse**: Text-Chunk- und Token-Verteilung  

### 3. Reference Analysis
- `extractAndAddRefrences.py`  
  **Rechtsnorm- & Case-Referenzen** extrahieren und anreichern  
- `addKnowledgeAboutRefrences.py`, `casesReferencesToId.py`  
  **Mapping & Vergleich** von Referenzen  
- `compareReferences.py`, `compareChainReferences.py`  
  **Verteilungs- und Kettenanalyse** von Verweisen  
- `plotRefFindingCapability*.py`  
  **Visualisierung**: Fähigkeit und Workflow der Referenzfindung  

### 4. HyDE & Chain of Verification
- `EvaluateHyDE_batched_chain_of_verification.py`  
- `chainOfVerificationEval.py`  
- `compareChainOfVerification.py`  
- `plotHyDEScores.py`  

### 5. Data Processing & Filtering
- `splitCases.py`  
- `filterCases.py`  
- `getData.py`  

### 6. Summarization & Q&A
- `summarizeTatbestand.py`  
- `AnswerAndDerive.py`, `AnswerAndDeriveCost.py`  

### 7. Indexing & Retrieval
- `createIndexBM25.py`  
- `evaluateTatBMFAISS.py`  
- `naiveRagWorkflow.py`  
- `advancedRagWorkflow.py` (optional)  

### 8. Visualization
- `createGraph*.py`  
- `similarityVisualization*.py`  
- `makeGraphsFromScores.py`  
- `plotCaseFrequency*.py`, `plotLawFrequency.py`, `plotTenor.py`  

### 9. Evaluation
- `evaluateGenerativeMetrics.py`  
- `llmCritique.py`  

### 10. Daten
- `enriched_cases_with_summaries.json`  

---

## ⚙️ Requirements

```text
python>=3.9
pip install -r requirements.txt
```

**Hinweis**: Viele Skripte nutzen `langchain`, `tiktoken`, `faiss` und `pydantic`.

---

## 🛠️ Usage Beispiele

### Einfacher Tenor-Predictor

```bash
python tenoranalyseUpdated.py   --input filtered_cases_10_percent.json   --output filtered_cases_10_percent_tenor.json
```

### Naive RAG Fallvorhersage

```bash
python naiveRagWorkflow.py   --input filtered_cases_10_percent.json   --output naive_rag_case_predictions.json
```

---

## 📈 Workflows & Architektur

1. **CasePredicitionSuperSimple austauschbares enrichment zu Standartisierung stets verwenden**  
   Standard LLM-basiert, dient als Baseline  
2. **Naive RAG**  
   Top-k ähnliche Tatbestände (BM25 & FAISS)  
3. **Advanced RAG**  
   Entscheidungsgründe HyDe CoVe lönnen reingeladen werden + Gerichtsbarkeits-Filter + LLM-Critique  

Jede Stufe steigert Vorhersagegenauigkeit und reduziert Bias.


Vielen Dank fürs Interesse!  
Bei Fragen oder Verbesserungsvorschlägen gern Issues öffnen.

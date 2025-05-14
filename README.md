# OptiDaDeGE
Optimierung der Datenanalyse von deutschen Gerichtsurteilen durch RAG mit LLMs

Legal Case Analysis Toolkit
This repository contains a suite of Python tools for analyzing, processing, and visualizing legal case data with a focus on German legal documents. The toolkit uses NLP, embeddings, and LLM techniques to extract insights and make predictions based on case details.

To fully replicate the examples in the Code you need to excecute the following steps
-get an openai_api_key and account and create a .env
-download cases.jsonl, laws.jsonl, book_codes.jsonl https://static.openlegaldata.io/dumps/de/
-excecute getData.py on cases.jsonl
-excecute extractAndAddRefrences.py 
-excecute generateEmbeddingsEntscheidungruende_threads.py, generateEmbeddingsTatbestand.py, createIndexBM25.py

This will provide you with the necessary retrieval data

The test data is found in enriched_cases_with_summaries.json
The json inputs in all files might need to be adjusted


Core Components
Case Analysis and Prediction
casePrediction.py - Core prediction model using OpenAI API to analyze legal cases and predict outcomes
casePredictionSuperSimple.py - Simplified version of the prediction model
casePredicitionReasoning.py - Enhanced version with reasoning capabilities
analysePredictions2.py - Statistical analysis of prediction results
Text Processing and Embeddings
generateEmbeddingsTatbestand.py - Creates embeddings from case facts ("Tatbestand")
generateEmbeddingsEntscheidungsgründe.py - Creates embeddings from decision rationales
generateEmbeddingsEntscheidungruende_threads.py - Multi-threaded version for better performance
countingChunks.py - Analyzes text chunk distribution
countingTokensTatbestände.py - Token counting for "Tatbestände" (case facts)
Reference Analysis
addKnwoledgeAboutRefrences.py - Adds contextual knowledge about legal references
casesRefrencesToId.py - Maps case references to unique identifiers
compareRefrences.py - Compares legal references across documents
compareChainRefrences.py - Analyzes references in chain format
extractAndAddRefrences.py - Extracts legal references and adds them to documents
extractAndAddRefrencesToQuestions.py - Adds references to legal questions
extractBookCodes.py - Extracts legal code references from documents
extractSimpleRefs.py - Simple reference extraction utility
plotRefFindingCapability.py - Visualizes reference detection capabilities
plotRefFindingCapabilityKs.py - Similar visualization with K-value focus
plotRefrences.py - General reference visualization tool
plotRefFindingAnswerWorkflow.py - Visualizes reference finding workflow
HyDE and Chain of Verification
EvaluateHyDE_batched_chain_of_verification.py - Evaluates Hypothetical Document Embeddings with verification chains
chainOfVerificationEval.py - Evaluates chain of verification approach
compareChainOfVerification.py - Compares different verification chain approaches
plotHyDEScores.py - Visualizes HyDE performance scores
Data Processing and Filtering
splitCases.py - Splits case data into manageable chunks
filterCases.py - Filters cases based on specific criteria
getData.py - Retrieves case data from sources
Summarization and Q&A
summarizeTatbestand.py - Creates summaries of case facts
AnswerAndDerive.py - Answers legal questions and derives insights
AnswerAndDeriveCost.py - Similar to above with cost analysis
Indexing and Retrieval
createIndexBM25.py - Creates BM25 index for document retrieval
evaluateTatBMFAISS.py - Evaluates FAISS vector database performance
naiveRagWorkflow.py - Simple RAG (Retrieval Augmented Generation) implementation
Visualization Tools
createGraph.py - Creates graph visualizations of legal relationships
createGraphSimple.py - Simplified version of graph creation
similarityVisualization.py - Visualizes similarity between documents
similarityVisualizationFilter.py - Filtered version of similarity visualization
makeGraphsFromScores.py - Generates graphs based on score data
plotCaseFrequency.py - Visualizes case frequency distributions
plotCaseFrequencyGraph.py - Graph-based visualization of case frequencies
plotCaseFrequencyZipf.py - Zipf's law analysis of case frequencies
plotLawFrequency.py - Visualizes law citation frequencies
plotTenor.py - Visualizes case tenor (outcome) distributions
Evaluation
evaluateGenerativeMetrics.py - Evaluates generative model performance
llmCritique.py - LLM-based critique of analysis results
Data Files
enriched_cases_with_summaries.json - Dataset containing processed cases with summaries
Requirements
Usage
Most scripts can be run directly with Python. For example:

Many scripts require input data files (like JSON case collections) and produce output files with analysis results.

Project Structure
The toolkit is organized around several primary workflows:

Case data processing and enrichment
Embedding generation and similarity analysis
Reference extraction and analysis
Prediction modeling and evaluation
Visualization of results and insights
Each script focuses on a specific task within these workflows, allowing for modular use depending on your analytical needs

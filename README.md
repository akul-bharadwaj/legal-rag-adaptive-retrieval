# Legal Agreement RAG with Adaptive Document-Aware Retrieval

This project builds and evaluates a Retrieval-Augmented Generation (RAG) pipeline for legal agreements such as non-disclosure agreements, annotated contracts, merger/acquisition agreements, and privacy-policy documents.

The goal is to process raw legal text files into a clean, chunked, searchable knowledge base and compare a baseline dense-retrieval RAG pipeline against an improved adaptive document-aware retrieval strategy.

---

## Project Overview

Legal documents are often long, clause-heavy, and contain repeated contractual language. A basic semantic retriever can retrieve legally similar clauses from the wrong document, which is risky in legal question-answering systems.

This project explores that challenge by:

- Loading and preprocessing a corpus of legal text files
- Performing exploratory analysis on document lengths, word frequency, and document similarity
- Chunking legal documents for retrieval
- Creating a Chroma vector database using OpenAI embeddings
- Building a baseline dense-retrieval RAG pipeline
- Evaluating the pipeline using RAGAS and retrieval metrics
- Implementing an adaptive document-aware retrieval strategy with BM25 and reranking
- Comparing baseline and adaptive retrieval performance

---

## Dataset Structure

The project expects the dataset to follow this structure:

```text
rag_legal/
├── corpus/
│   ├── contractnli/
│   ├── cuad/
│   ├── maud/
│   └── privacy_qa/
│
├── benchmark/
│   ├── contractnli.json
│   ├── cuad.json
│   └── maud.json
│
├── RAG_Legal_Docs.ipynb
├── requirements.txt
└── README.md
```

The corpus contains legal documents in `.txt` format. The benchmark folder contains JSON files with benchmark questions, answers, and source snippets.

---

## Key Features

### 1. Legal Text Preprocessing

The notebook performs text cleaning while preserving important legal meaning. For example, legal terms such as `not`, `shall`, `may`, and `must` are not removed from the RAG-ready text because they can change the interpretation of a clause.

Two text versions are created:

- **RAG-ready text**: cleaned while preserving legal meaning
- **Analysis text**: cleaned further for word-frequency and exploratory analysis

---

### 2. Exploratory Data Analysis

The notebook includes analysis of:

- Average, maximum, and minimum document length
- Most common and least common words
- TF-IDF similarity between documents
- Similarity comparison between the first 10 documents and 10 randomly selected documents

These steps help understand the structure and repetitiveness of legal agreements before building the retrieval system.

---

### 3. Chunking and Vector Database

Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter`.

The chunks are embedded using OpenAI embeddings and stored in a Chroma vector database.

This enables semantic similarity search across the legal document corpus.

---

## RAG Pipelines

### Baseline RAG

The baseline pipeline uses:

- Dense vector retrieval from Chroma
- OpenAI embeddings
- LangChain LCEL chain
- Prompt-based legal question answering
- RAGAS evaluation

The baseline retrieves the top relevant chunks globally from the vector database.

---

### Adaptive Document-Aware RAG

The improved pipeline adds an adaptive retrieval strategy.

Instead of always retrieving globally, it first checks whether the query contains strong document-level signals such as party names or agreement names.

If document match confidence is high, the pipeline performs document-aware candidate filtering, BM25 retrieval within candidate documents, and cross-encoder reranking.

If document match confidence is low, it falls back to the baseline dense retriever.

This avoids the failure mode of hard document routing, where the system may restrict retrieval to the wrong document.

---

## Evaluation

The pipelines were evaluated using 10 randomly sampled benchmark questions.

### RAGAS Evaluation

| Metric | Baseline RAG | Adaptive Document-Aware RAG |
|---|---:|---:|
| Faithfulness | 0.3133 | 0.3667 |
| Context Precision | 0.7750 | 0.6306 |
| Context Recall | 0.5500 | 0.6000 |

### Retrieval Evaluation

| Metric | Baseline RAG | Adaptive Document-Aware RAG |
|---|---:|---:|
| Precision@4 | 0.625 | 0.725 |
| Recall@4 | 0.800 | 1.000 |

---

## Key Insights

The adaptive document-aware pipeline improved retrieval coverage and source matching.

Precision@4 improved from **0.625 to 0.725**, and Recall@4 improved from **0.800 to 1.000**. This shows that the adaptive approach was better at retrieving the expected source documents within the top 4 results.

Faithfulness also improved from **0.3133 to 0.3667**, and context recall improved from **0.5500 to 0.6000**.

However, context precision decreased from **0.7750 to 0.6306**. This suggests that the adaptive pipeline retrieved a broader set of useful contexts, but some retrieved chunks were less focused.

For legal RAG systems, high recall is especially important because missing a relevant clause or agreement can lead to incomplete or unreliable answers. The adaptive pipeline therefore provides a useful improvement, while leaving room for further optimisation in context precision.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- LangChain
- ChromaDB
- OpenAI embeddings and chat models
- RAGAS
- BM25 retrieval
- SentenceTransformers cross-encoder reranking
- ROUGE score package

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/legal-rag-adaptive-retrieval.git
cd legal-rag-adaptive-retrieval
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Do not commit the `.env` file to GitHub.

### 5. Run the notebook

Open the notebook:

```bash
jupyter notebook RAG_Legal_Docs_GitHub_Cleaned.ipynb
```

Run the cells sequentially.

---

## Environment Variables

The project uses the following environment variable:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

A `.gitignore` file should include:

```text
.env
__pycache__/
.ipynb_checkpoints/
legal_agreement_vector_db/
chroma_db/
```

---

## Limitations

This project is an experimental legal RAG pipeline and is not intended to provide legal advice.

Current limitations include:

- Evaluation was performed on a small sample of benchmark questions
- Context precision dropped in the adaptive pipeline
- Chunking is character-based rather than clause-aware
- Retrieval can still return legally similar but source-incorrect content
- RAGAS scores may vary depending on LLM evaluator settings
- The pipeline depends on external API calls for embeddings and generation

---

## Future Improvements

Potential improvements include:

- Adding Reciprocal Rank Fusion to combine dense and BM25 rankings more systematically
- Improving cross-encoder reranking to increase context precision
- Using metadata-based filtering for dataset type or document category
- Trying legal-specific chunking strategies based on clause boundaries
- Strengthening the answer-generation prompt to improve faithfulness
- Improving document-match confidence scoring
- Evaluating on a larger benchmark sample

---

## Disclaimer

This project is for educational and portfolio purposes only. It should not be used as a substitute for professional legal advice.

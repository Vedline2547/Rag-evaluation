# RAG Evaluation using RAGAS, BLEU, ROUGE & BERTScore

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline using a Power BI PDF document as its knowledge source. It combines LangChain, FAISS, Hugging Face embeddings, and Ollama to retrieve relevant information and generate context-aware answers.

The project focuses on evaluating the quality of generated responses using multiple evaluation techniques, including RAGAS, BLEU, ROUGE, and BERTScore.

The objective is to understand how well a RAG system retrieves relevant information, generates accurate responses, and maintains consistency with reference answers.

## Technologies & Libraries

- **Python** – Core programming language.
- **LangChain** – RAG pipeline development and orchestration.
- **Ollama (Gemma 3 1B)** – Local language model for answer generation.
- **FAISS** – Vector database for similarity search.
- **Hugging Face Embeddings** – Text embedding generation using `all-MiniLM-L6-v2`.
- **RAGAS** – RAG-specific evaluation framework.
- **NLTK** – BLEU score calculation.
- **ROUGE** – Text overlap evaluation.
- **BERTScore** – Semantic similarity evaluation.
- **Pandas** – Data handling and evaluation results management.

## RAG Architecture

1. **Document Loading:** Load the Power BI PDF using LangChain's PyPDFLoader.
2. **Text Splitting:** Divide the document into smaller chunks using RecursiveCharacterTextSplitter.
3. **Embedding Generation:** Convert text chunks into numerical representations using Hugging Face embeddings.
4. **Vector Storage:** Store embeddings in a FAISS vector database.
5. **Retrieval:** Retrieve relevant document chunks based on user queries.
6. **Response Generation:** Use Gemma 3 1B through Ollama to generate answers using retrieved context.
7. **Evaluation:** Compare generated answers against reference answers using different evaluation metrics.

## Evaluation Metrics

### 1. RAGAS

Used to evaluate the quality of the RAG pipeline through:

- **Context Recall:** Measures whether retrieved contexts contain the information needed to answer questions.
- **Faithfulness:** Measures whether generated answers are supported by retrieved contexts.
- **Factual Correctness:** Measures factual agreement between generated answers and reference answers.

### 2. BLEU Score

Measures n-gram precision by evaluating the overlap between generated answers and reference answers.

### 3. ROUGE Score

Measures textual overlap between generated and reference answers. This project uses ROUGE-1 F1.

### 4. BERTScore

Uses contextual embeddings from transformer models to measure semantic similarity between generated answers and reference answers.

## Project Structure

```text
RAG-Evaluation/
│
├── Evaluation.py
├── Power+BI+Ebook.pdf
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Vedline2547/Rag-evaluation.git
```

Navigate into the project directory:

```bash
cd RAG-Evaluation
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install the required dependencies:

```bash
pip install langchain langchain-community langchain-core langchain-classic langchain-ollama langchain-huggingface langchain-text-splitters faiss-cpu sentence-transformers pypdf ragas rouge nltk transformers bert-score torch pandas
```

Install and configure Ollama, then download the model:

```bash
ollama pull gemma3:1b
```

## Running the Project

Run the Python script:

```bash
python Evaluation.py
```

The script loads the PDF, builds the vector database, generates responses, and evaluates the results.

## Expected Outputs

The evaluation produces:

- RAGAS evaluation scores.
- BLEU scores for generated responses.
- ROUGE-1 F1 scores.
- BERTScore F1 scores.
- A CSV file containing evaluation results.

## Key Learning Outcomes

- Understanding Retrieval-Augmented Generation architecture.
- Implementing semantic search using FAISS.
- Integrating local LLMs into RAG applications.
- Evaluating retrieval quality and generated responses.
- Understanding the differences between lexical and semantic evaluation metrics.
- Comparing multiple evaluation approaches for RAG systems.

## Future Improvements

- Expand the evaluation dataset with more diverse questions.
- Experiment with different embedding models.
- Compare different local LLMs.
- Implement proper BARTScore evaluation.
- Optimize chunk size and retrieval parameters.
- Build a Streamlit interface for interactive RAG evaluation.
- Visualize and compare evaluation metrics.

## Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning Enthusiast | Python Developer

GitHub: [@Vedline2547](https://github.com/Vedline2547)

---

*This project explores practical approaches to evaluating Retrieval-Augmented Generation systems using both traditional NLP metrics and modern semantic evaluation techniques.*

# TrialLens – AI-Powered Clinical Trial Discovery and Eligibility Assistant

TrialLens is an educational AI application designed to help users explore clinical trials and understand published eligibility criteria using Retrieval-Augmented Generation (RAG).

The application retrieves relevant clinical trial information from ClinicalTrials.gov and uses a language model to generate answers grounded in retrieved trial information.

## Project Overview

Clinical trial information can be extensive and difficult to navigate. TrialLens aims to make trial discovery and eligibility information easier to explore through semantic search and natural-language questions.

## Key Features

- **Clinical Trial Search:** Search for clinical trials using natural-language queries.
- **RAG-Based Question Answering:** Generate answers using retrieved clinical trial information.
- **Eligibility Q&A:** Ask questions about inclusion and exclusion criteria using a ClinicalTrials.gov NCT ID.
- **Patient Profile Comparison:** Compare a synthetic patient profile against published trial eligibility criteria.
- **Vector Search:** Retrieve relevant information using FAISS and Sentence Transformers.
- **Evaluation:** Includes retrieval and answer-evaluation datasets and scripts.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Data Source | ClinicalTrials.gov API |
| Data Processing | Pandas, NumPy |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Large Language Model | Groq – openai/gpt-oss-20b |
| Environment Management | python-dotenv |

## System Architecture

1. **Data Collection:** Retrieve clinical trial records from ClinicalTrials.gov.
2. **Data Preprocessing:** Clean and prepare the collected trial information.
3. **Document Chunking:** Split trial information into section-based chunks.
4. **Embedding Generation:** Convert text chunks into numerical embeddings.
5. **Vector Database:** Store embeddings in a FAISS index.
6. **Retrieval:** Retrieve relevant clinical trial information based on user queries.
7. **Generation:** Use the Groq language model to generate responses based on retrieved evidence.
8. **Streamlit Interface:** Provide an interactive application for users.

## Project Structure

```text
TrialLens/
│
├── app.py
├── data_collection.py
├── data_preprocessing.py
├── document_chunking.py
├── embedding_generation.py
├── vector_database.py
├── retrieval.py
├── rag_pipeline.py
├── eligibility_retrieval.py
├── evaluation.py
├── retrieval_evaluation.py
├── requirements.txt
├── .gitignore
│
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shahana2421/TrialLens.git
cd TrialLens
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never upload your API key or `.env` file to GitHub.

### 5. Prepare the data and vector database

Run the data pipeline scripts in this order:

```bash
python data_collection.py
python data_preprocessing.py
python document_chunking.py
python embedding_generation.py
python vector_database.py
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

Note: The generated datasets, embeddings and FAISS index are not included in this GitHub repository. They must be generated locally before running the application.

## Evaluation

The project includes preliminary evaluation scripts and datasets for:

- Retrieval performance
- Retrieval relevance
- RAG answer evaluation
- Claim-level evaluation

These evaluations are exploratory and have not established clinical accuracy or clinical reliability.

## Evaluation Results

TrialLens includes preliminary experiments to evaluate clinical trial retrieval performance.

### Retrieval Evaluation

The initial evaluation was conducted using six manually created search queries.

| Metric | Result |
|---|---:|
| Precision@5 | 80% |
| Recall@5 | 63.19% |
| Mean Reciprocal Rank (MRR@5) | 1.00 |

### Evaluation Methodology

- Six search queries were used for the initial evaluation.
- Relevance labels were manually assigned.
- Recall was calculated using a manually judged top-10 candidate pool, not the entire dataset.
- The evaluation was conducted on a clinical trial dataset containing 295 unique records across six medical conditions.

### Limitations

- The evaluation dataset is small and may not represent all user queries.
- Manual relevance judgments may introduce subjective bias.
- Recall may not capture relevant trials outside the judged candidate pool.
- A keyword-based baseline comparison, such as BM25, has not yet been conducted.
- Further evaluation is required for answer quality and patient-profile comparison.

These results are preliminary and should not be interpreted as clinical accuracy or clinical validation.

## Important Disclaimer

TrialLens is an educational prototype developed for learning and portfolio demonstration.

- It is not a medical diagnostic tool.
- It does not provide medical advice or treatment recommendations.
- Patient profile comparisons are preliminary comparisons against published eligibility criteria, not actual clinical eligibility decisions.
- Missing patient information may prevent assessment of individual criteria.
- Trial information may change after collection.
- Generated answers may contain inaccuracies or omit relevant information.

Always consult the original ClinicalTrials.gov record and qualified healthcare professionals for clinical decisions.

## Future Improvements

- Improve retrieval relevance and evidence grounding.
- Expand clinical trial coverage.
- Improve evaluation with larger, independently reviewed datasets.
- Improve handling of missing patient information.
- Enhance the user interface and source verification.

## Author

**Shahana S**

GitHub: https://github.com/Shahana2421

---

*Built as an educational project to explore Retrieval-Augmented Generation, semantic search, and AI-powered information retrieval.*

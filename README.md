# AI Document Analyser

AI-powered document analysis web application built with FastAPI, Hugging Face, LangChain, and FAISS.

## Project Overview

This project enables users to upload PDF documents and ask natural language questions about their content. It combines PDF parsing, semantic embedding, vector search, and question-answering to deliver fast, accurate responses from documents.

## Key Features

- PDF upload and text extraction using `pypdf`
- Semantic search with `sentence-transformers/all-MiniLM-L6-v2`
- Vector database built with `FAISS`
- Question-answering through Hugging Face inference API
- Resume-friendly use cases:
  - Document Q&A assistant for resumes, reports, SOPs, and research papers
  - Knowledge retrieval from unstructured PDF content
  - Proof of concept for retrieval-augmented generation (RAG)

## Tech Stack

- Python
- FastAPI
- Uvicorn
- LangChain
- FAISS
- Hugging Face Hub
- Sentence Transformers
- PDF parsing with `pypdf`
- Frontend served with Jinja2 templates

## Architecture

1. User uploads a PDF via `/upload`
2. PDF text is extracted and split into chunks
3. Sentence embeddings are generated and indexed in a FAISS vector store
4. User submits a question via `/ask`
5. The application retrieves the most relevant document chunks and generates an answer using Hugging Face question-answering

## Getting Started

### Prerequisites

- Python 3.11 or newer
- A Hugging Face access token with inference permissions

### Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with the following variables:

```env
HF_TOKEN=your_huggingface_token
HF_MODEL_ID=deepset/roberta-base-squad2
HF_PROVIDER=hf-inference
```

### Run Locally

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## Usage

- Upload a PDF document from the web interface
- Ask questions like:
  - `What skills are listed?`
  - `Who is the contact person?`
  - `Summarize the main points from this document.`

## Next Improvements

- Add support for multiple document uploads and session-based vector stores
- Improve UI/UX for question entry and response display
- Add search history and conversation context
- Support additional file formats such as DOCX and TXT

## Notes

- The project currently maintains uploaded PDFs in the `uploads/` folder.
- `.env` is intentionally excluded from version control to keep API keys secure.

# Resume Parser & Extractor

A high-performance, async microservice that parses resumes into structured data (JSON) and allows you to ask questions about candidates using semantic search. Built with **FastAPI**, **LangGraph**, **ChromaDB**, and **Google Gemini**.

## 🚀 Features

- **Structured Parsing**: Extracts Name, Email, Skills, Experience, and Education into strict JSON using Gemini 1.5 Flash.
- **Smart Q&A**: Ask questions like *"Does this candidate have leadership experience?"* grounded in the resume text.
- **High Concurrency**: Supports batch processing of 100+ resumes simultaneously using `asyncio` and Semaphores.
- **Stateful Workflow**: Uses **LangGraph** to manage ingestion, parsing, and retrieval states.
- **Vector Search**: Uses **ChromaDB** for fast semantic lookup (local & efficient).

## 🛠️ Tech Stack

- **Framework**: FastAPI (Async Python)
- **Orchestration**: LangGraph + LangChain
- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: Google Text Embeddings 004
- **Database**: ChromaDB (Vector Store)

## 📦 Setup

### 1. Clone & Install
```bash
git clone <repo-url>
cd resume-parser
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory:
```bash
touch .env
```
Add your Google API Key:
```ini
GOOGLE_API_KEY=your_google_api_key_here
```
*(Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey))*

### 3. Run the Server
```bash
uvicorn app.main:app --reload
```
The API will start at `http://localhost:8000`.

## 📖 Usage

### Swagger UI
Open **[http://localhost:8000/docs](http://localhost:8000/docs)** to test endpoints interactively.

### 1. Upload a Resume (Single)
**POST** `/upload`
- **Body**: `multipart/form-data` (file: resume.pdf)
- **Response**: Structured JSON (Skills, Experience, etc.)

### 2. Batch Upload (Concurrent)
**POST** `/batch-upload`
- **Body**: List of files
- **Behavior**: Processes multiple PDFs in parallel (limited by Semaphore to avoid rate limits).

### 3. Ask a Question
**POST** `/query`
```json
{
  "question": "Does this candidate have experience with Python and AWS?"
}
```

## 🧪 Testing

Run the benchmark test to simulate processing **100 concurrent resumes** (using mocks):

```bash
pytest tests/test_benchmark.py -v -s
```

## 🏗️ Architecture

1.  **Ingestion**: `PyPDFLoader` reads the PDF → `RecursiveCharacterTextSplitter` chunks it.
2.  **Embedding**: Chunks are embedded via `text-embedding-004` and stored in `ChromaDB`.
3.  **Parsing**: `gemini-1.5-flash` with `with_structured_output` extracts the `ResumeData` Pydantic model.
4.  **Retrieval**: `ask_question` performs a similarity search in ChromaDB and feeds relevant chunks to the LLM for the answer.

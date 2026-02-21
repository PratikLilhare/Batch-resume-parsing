from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from pydantic import BaseModel
import shutil
import os
import asyncio
from dotenv import load_dotenv

from app.helpers import ingest_pdf, parse_resume, ask_question, ResumeState
from app.models import ResumeData

load_dotenv()

app = FastAPI(title="Resume RAG Parser")

# Limit concurrency to 5 parallel resumes to avoid hitting Gemini rate limits
# If you have a higher tier API key, you can increase this number.
CONCURRENCY_LIMIT = 5
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

class QueryRequest(BaseModel):
    question: str

async def process_single_resume(file: UploadFile) -> ResumeData:
    """Helper function to process one resume with semaphore protection."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")

    temp_path = f"temp_{file.filename}"
    
    # Save file (blocking IO, run in thread if file is huge, but usually fine here)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        async with semaphore:  # Wait for a slot
            # Initial state
            state = {
                "pdf_path": temp_path,
                "text_chunks": [],
                "parsed_data": None,
                "question": "",
                "answer": ""
            }
            
            # Run Ingest (Async)
            ingest_result = await ingest_pdf(state)
            state.update(ingest_result)
            
            # Run Parse (Async)
            parse_result = await parse_resume(state)
            
            return parse_result["parsed_data"]
            
    except Exception as e:
        print(f"Error processing {file.filename}: {e}")
        # In a real app, return an error object instead of crashing
        raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/upload", response_model=ResumeData)
async def upload_resume(file: UploadFile = File(...)):
    """Uploads a single PDF resume (Async)."""
    return await process_single_resume(file)

@app.post("/batch-upload", response_model=List[ResumeData])
async def batch_upload_resumes(files: List[UploadFile] = File(...)):
    """
    Uploads up to 100 PDF resumes and processes them concurrently.
    Concurrency is limited by a Semaphore to respect API rate limits.
    """
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Batch limit exceeded (max 100 files)")
    
    # Run all tasks concurrently
    tasks = [process_single_resume(file) for file in files]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out failures or re-raise
    final_results = []
    for res in results:
        if isinstance(res, Exception):
            # For simplicity, we skip failed files or you could return partial errors
            print(f"Batch processing error: {res}")
        else:
            final_results.append(res)
            
    return final_results

@app.post("/query")
async def query_resume(request: QueryRequest):
    """Ask a question about the uploaded resume(s) using RAG."""
    try:
        state = {
            "pdf_path": "", 
            "text_chunks": [], 
            "parsed_data": None, 
            "question": request.question, 
            "answer": ""
        }
        result = await ask_question(state)
        return {"answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

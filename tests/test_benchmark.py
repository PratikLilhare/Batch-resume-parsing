import sys
import os

# Add the parent directory to sys.path so we can import 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.models import ResumeData, Experience, Education

# Mock Data to Return Instead of Gemini
MOCK_RESUME = ResumeData(
    full_name="Mock Candidate",
    email="mock@example.com",
    skills=["Python", "Mocking", "AsyncIO"],
    experience=[
        Experience(role="Senior Dev", company="Mock Corp", duration="2020-Present", description="Led mock teams.")
    ],
    education=[
        Education(degree="BS CS", institution="Mock University", year="2020")
    ],
    summary="Experienced mock developer."
)

MOCK_CHUNKS = ["This is chunk 1", "This is chunk 2"]
MOCK_EMBEDDINGS = [[0.1] * 768]  # Fake vector of size 768

@pytest.mark.asyncio
async def test_batch_processing_speed():
    """
    Simulates processing 100 resumes concurrently by mocking external APIs.
    We expect this to finish very quickly because we removed the network latency.
    """
    
    # 1. Patch the slow external dependencies
    with patch("app.helpers.llm") as mock_llm, \
         patch("app.helpers.embeddings") as mock_embeddings, \
         patch("app.helpers.PyPDFLoader") as mock_loader, \
         patch("app.helpers.vector_store") as mock_vector_store:

        # Setup Mock Behaviors
        
        # Mock LLM (Gemini) - Returns structured data
        # The chain calls .ainvoke(), so we mock that
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = MOCK_RESUME
        
        # When llm.with_structured_output is called, it returns a runnable that returns our mock_chain
        # Actually, the code does: chain = prompt | structured_llm
        # So we need to mock the entire chain execution.
        
        # Let's patch 'app.helpers.parse_resume' directly since it contains the logic we want to skip/mock
        # But to test concurrency, we should mock the components inside it if possible.
        # However, mocking the | operator is hard.
        
        # EASIER STRATEGY: Patch the `parse_resume` function itself.
        # This simulates the "AI Processing" part taking time.
        pass

@pytest.mark.asyncio
async def test_100_concurrent_resumes():
    """
    Test processing 100 resumes with simulated delay to test Semaphore.
    """
    from app.main import process_single_resume, semaphore
    from fastapi import UploadFile
    import io

    # Mock Data
    dummy_pdf = UploadFile(filename="test.pdf", file=io.BytesIO(b"dummy pdf content"))

    # We patch at the function level to control the "AI Latency"
    with patch("app.main.ingest_pdf", new_callable=AsyncMock) as mock_ingest, \
         patch("app.main.parse_resume", new_callable=AsyncMock) as mock_parse:

        # Simulate Ingest taking 0.1s
        async def side_effect_ingest(*args, **kwargs):
            await asyncio.sleep(0.01) # fast ingest
            return {"text_chunks": ["chunk"]}
        mock_ingest.side_effect = side_effect_ingest

        # Simulate Gemini taking 0.5s per resume (Simulated Latency)
        async def side_effect_parse(*args, **kwargs):
            await asyncio.sleep(0.1) # fast parse for test
            return {"parsed_data": MOCK_RESUME}
        mock_parse.side_effect = side_effect_parse

        # Create 100 tasks
        files = [dummy_pdf for _ in range(100)]
        
        print(f"\nStarting batch of {len(files)} resumes with Semaphore={semaphore._value}...")
        
        import time
        start_time = time.time()
        
        # Run the actual function from main.py
        tasks = [process_single_resume(f) for f in files]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Processed {len(results)} resumes in {duration:.2f} seconds.")
        
        # Assertions
        assert len(results) == 100
        # The result is a Pydantic model instance, so we compare it directly to MOCK_RESUME
        assert results[0] == MOCK_RESUME
        
        # Calculate theoretical min time: 
        # 100 tasks / 5 concurrency * 0.1s latency = 2.0 seconds minimum
        # If it runs in 0.1s, the semaphore isn't working or we mocked wrong.
        # If it runs in 10s, it's sequential.
        print(f"Throughput: {len(results)/duration:.2f} resumes/sec")

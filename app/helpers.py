import os
import asyncio
from dotenv import load_dotenv
from typing import List, TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from app.models import ResumeData

load_dotenv()

# Initialize Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Initialize Vector Store (ChromaDB)
PERSIST_DIRECTORY = "./chroma_db"
vector_store = Chroma(
    collection_name="resumes",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY
)

class ResumeState(TypedDict):
    """State for the LangGraph workflow."""
    pdf_path: str
    text_chunks: List[str]
    parsed_data: ResumeData
    question: str
    answer: str

async def ingest_pdf(state: ResumeState):
    """Loads PDF, splits text, and stores embeddings (Async)."""
    # PyPDFLoader is blocking, run in thread
    def _load():
        loader = PyPDFLoader(state["pdf_path"])
        return loader.load()
    
    pages = await asyncio.to_thread(_load)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    
    # Chroma add_documents is blocking, run in thread
    await asyncio.to_thread(vector_store.add_documents, documents=splits)
    
    return {"text_chunks": [doc.page_content for doc in splits]}

async def parse_resume(state: ResumeState):
    """Extracts structured data using Gemini (Async)."""
    context = "\n".join(state["text_chunks"])
    
    structured_llm = llm.with_structured_output(ResumeData)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert resume parser. Extract structured information from the following resume text."),
        ("human", "{context}")
    ])
    
    chain = prompt | structured_llm
    # Use ainvoke for async LLM call
    result = await chain.ainvoke({"context": context})
    
    return {"parsed_data": result}

async def ask_question(state: ResumeState):
    """Answers questions using RAG (Async)."""
    retriever = vector_store.as_retriever()
    relevant_docs = await retriever.ainvoke(state["question"])
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based only on the following context:\n\n{context}"),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    response = await chain.ainvoke({"context": context, "question": state["question"]})
    
    return {"answer": response.content}

# Build the Graph
workflow = StateGraph(ResumeState)

workflow.add_node("ingest", ingest_pdf)
workflow.add_node("parse", parse_resume)
workflow.add_node("rag_query", ask_question)

# Flow: ingest -> parse -> END
workflow.add_edge(START, "ingest")
workflow.add_edge("ingest", "parse")
workflow.add_edge("parse", END)

# Compile the graph
app_graph = workflow.compile()

"""
AI PDF & ATS Assistant - FastAPI Web Application
Integrated with Model Context Protocol (MCP)
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure stdout/stderr utf-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure silence of noisy telemetry/progress bars
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Load environment
load_dotenv(BASE_DIR / ".env")

UPLOAD_DIR = os.getenv(
    "UPLOAD_DIR",
    "/tmp/pdf-assistant/uploads" if os.getenv("VERCEL") else "data/uploads"
)

# Core imports
from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import get_vector_store, DB_ROOT
from src.retriever import get_retriever
from src.chatbot import get_llm
from src.summarizer import summarize_document
from src.assistant import handle_question
from src.file_hash import calculate_file_hash
from src.ats.parser import extract_pdf_text
from src.ats.analyzer import analyze_resume

# MCP Server Tools integration
from mcp_server import (
    ask_pdf as mcp_ask_pdf,
    summarize_pdf as mcp_summarize_pdf,
    search_web_context as mcp_search_web_context,
    analyze_resume_ats as mcp_analyze_resume_ats,
    calculate_ats_score_metric as mcp_calculate_ats_score_metric,
    extract_pdf_raw_text as mcp_extract_pdf_raw_text,
    list_available_documents as mcp_list_available_documents,
)

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="AI PDF Assistant (MCP Integrated)",
    description="Chat with PDF documents and analyze ATS Resumes using Model Context Protocol (MCP)",
    version="2.1.0"
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

# --------------------------------------------------
# GLOBAL STATE & INITIALIZATION
# --------------------------------------------------

llm = get_llm()
documents_store = {}
chat_history = []


def _init_default_documents():
    """Auto-register default sample PDF into documents_store."""
    data_dir = BASE_DIR / "data"
    sample = data_dir / "AI_Research.pdf"
    if sample.exists():
        try:
            filename = sample.name
            docs = load_pdf(str(sample))
            chunks = split_documents(docs)
            for chunk in chunks:
                chunk.metadata["filename"] = filename
            emb = get_embeddings()
            vs = get_vector_store(str(sample), chunks, emb)
            retriever = get_retriever(vs, chunks)
            documents_store[filename] = {
                "retriever": retriever,
                "chunks": chunks,
                "summary": None,
                "file_path": str(sample),
            }
            print(f"[Init] Ready with document: {filename} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"[Init] Warning preloading {sample.name}: {e}")


# Initialize default repository documents
_init_default_documents()


# --------------------------------------------------
# HOME & PAGES
# --------------------------------------------------

@app.get("/")
def home():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/ats-analysis")
def ats_analysis_page():
    return FileResponse(str(BASE_DIR / "static" / "ats.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mcp_enabled": True,
        "mcp_tools_count": 7,
        "active_documents": list(documents_store.keys())
    }


@app.get("/mcp/tools")
def get_mcp_tools():
    """Return all available Model Context Protocol tools and their capabilities."""
    return {
        "protocol": "Model Context Protocol (MCP) 2024-11-05",
        "server": "PDF-ATS-Assistant",
        "tools": [
            {
                "name": "ask_pdf",
                "description": "Hybrid Semantic (Chroma) + BM25 search with source page citations.",
            },
            {
                "name": "summarize_pdf",
                "description": "Hierarchical executive multi-chunk summarization.",
            },
            {
                "name": "search_web_context",
                "description": "Live web search verification and grounding via Tavily.",
            },
            {
                "name": "analyze_resume_ats",
                "description": "Deterministic ATS resume scoring, skills gap, and keyword analysis.",
            },
            {
                "name": "calculate_ats_score_metric",
                "description": "Quick calculation of match score and missing skill metrics.",
            },
            {
                "name": "extract_pdf_raw_text",
                "description": "Plain text extraction from any uploaded PDF.",
            },
            {
                "name": "list_available_documents",
                "description": "List all PDF documents across repository folders.",
            },
        ]
    }


# --------------------------------------------------
# DOCUMENTS LIST
# --------------------------------------------------

@app.get("/documents")
def get_documents():
    return {
        "documents": list(documents_store.keys()),
        "mcp_available": mcp_list_available_documents()
    }


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global documents_store, chat_history

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_filename = os.path.basename(file.filename or "upload.pdf")
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not save the uploaded PDF.") from exc

    try:
        documents = load_pdf(file_path)
        chunks = split_documents(documents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read or parse PDF text.") from exc

    if not chunks:
        try:
            os.remove(file_path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="PDF has no selectable text.")

    for chunk in chunks:
        chunk.metadata["filename"] = file.filename

    try:
        embedding_model = get_embeddings()
        vector_store = get_vector_store(file_path, chunks, embedding_model)
        document_retriever = get_retriever(vector_store, chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Index build error: {exc}") from exc

    documents_store[file.filename] = {
        "retriever": document_retriever,
        "chunks": chunks,
        "summary": None,
        "file_path": file_path
    }

    chat_history = []
    return {
        "message": "PDF uploaded and processed with MCP index successfully",
        "filename": file.filename,
        "pages": len(documents),
        "chunks": len(chunks),
        "documents": list(documents_store.keys()),
        "mcp_ready": True
    }


# --------------------------------------------------
# CHAT (MCP INTEGRATED)
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    selected_document: str = ""
    use_web: bool = False
    search_web: bool = False


def is_summary_request(question: str) -> bool:
    summary_phrases = [
        "summarize",
        "summary",
        "what is this pdf about",
        "what is the pdf about",
        "what is this document about",
        "what is the document about",
        "overview of the pdf",
        "overview of the document"
    ]
    q = question.lower()
    return any(p in q for p in summary_phrases)


@app.post("/chat")
def chat(request: ChatRequest):
    global chat_history

    # 1. Live Web Search via MCP tool
    if request.use_web:
        mcp_res = mcp_search_web_context(request.question)
        answer_text = f"⚡ [MCP Tool: search_web_context]\n\n" + mcp_res.get("answer", "")
        chat_history.append(f"User: {request.question}")
        chat_history.append(f"Assistant: {answer_text}")
        return {
            "answer": answer_text,
            "route": "web",
            "sources": mcp_res.get("sources", []),
            "mcp_tool": "search_web_context"
        }

    # 2. PDF Document Check
    if not request.selected_document or request.selected_document not in documents_store:
        return {
            "error": "Please select a valid PDF first or enable Web Search."
        }

    selected_pdf = documents_store[request.selected_document]
    doc_path = selected_pdf.get("file_path", request.selected_document)

    # 3. Summarization via MCP tool
    if is_summary_request(request.question):
        if selected_pdf["summary"] is None:
            summary = mcp_summarize_pdf(doc_path)
            selected_pdf["summary"] = summary
        else:
            summary = selected_pdf["summary"]

        answer_text = f"⚡ [MCP Tool: summarize_pdf]\n\n" + summary
        chat_history.append(f"User: {request.question}")
        chat_history.append(f"Assistant: {answer_text}")
        return {
            "answer": answer_text,
            "route": "pdf",
            "sources": [{"filename": request.selected_document}],
            "mcp_tool": "summarize_pdf"
        }

    # 4. Standard Question Answering via MCP Hybrid RAG tool
    rag_answer = mcp_ask_pdf(doc_path, request.question)
    answer_text = f"⚡ [MCP Tool: ask_pdf]\n\n" + rag_answer

    chat_history.append(f"User: {request.question}")
    chat_history.append(f"Assistant: {answer_text}")

    return {
        "answer": answer_text,
        "route": "pdf",
        "sources": [{"filename": request.selected_document}],
        "mcp_tool": "ask_pdf"
    }


# --------------------------------------------------
# ATS ANALYSIS (MCP INTEGRATED)
# --------------------------------------------------

@app.post("/ats/analyze")
async def ats_analyze(
    resume: UploadFile = File(...),
    job_description: UploadFile = File(...)
):
    """Analyze resume against job description using MCP analyze_resume_ats."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Resume must be a PDF file.")
    if not job_description.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Job description must be a PDF file.")

    resume_filename = "ats_res_" + os.path.basename(resume.filename)
    jd_filename = "ats_jd_" + os.path.basename(job_description.filename)

    resume_path = os.path.join(UPLOAD_DIR, resume_filename)
    jd_path = os.path.join(UPLOAD_DIR, jd_filename)

    try:
        with open(resume_path, "wb") as buffer:
            shutil.copyfileobj(resume.file, buffer)
        with open(jd_path, "wb") as buffer:
            shutil.copyfileobj(job_description.file, buffer)

        # Call official FastMCP Tool
        mcp_result = mcp_analyze_resume_ats(
            resume_path=resume_path,
            job_description_path=jd_path
        )

        return {
            "overall_score": mcp_result["overall_score"],
            "match_level": mcp_result["match_level"],
            "analysis": {
                "matching_skills": mcp_result["matching_skills"],
                "missing_skills": mcp_result["missing_skills"],
                "matching_keywords": mcp_result["matching_keywords"],
                "missing_keywords": mcp_result["missing_keywords"],
                "experience_match": mcp_result["experience_match"],
                "strengths": mcp_result["strengths"],
                "recommendations": mcp_result["recommendations"],
            },
            "mcp": {
                "tool": "analyze_resume_ats",
                "protocol": "Model Context Protocol (MCP) 2024-11-05",
                "status": "success"
            }
        }
    except Exception as e:
        print(f"ATS MCP analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"ATS analysis error: {str(e)}")
    finally:
        for p in [resume_path, jd_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


@app.post("/ats/sample")
def ats_sample():
    """Quickly run MCP ATS analysis on repository sample files."""
    sample_resume = str(BASE_DIR / "data" / "ats" / "resume.pdf")
    sample_jd = str(BASE_DIR / "data" / "ats" / "job_description.pdf")

    if not os.path.exists(sample_resume) or not os.path.exists(sample_jd):
        raise HTTPException(status_code=404, detail="Sample ATS PDFs not found in data/ats/")

    mcp_result = mcp_analyze_resume_ats(
        resume_path=sample_resume,
        job_description_path=sample_jd
    )

    return {
        "overall_score": mcp_result["overall_score"],
        "match_level": mcp_result["match_level"],
        "analysis": {
            "matching_skills": mcp_result["matching_skills"],
            "missing_skills": mcp_result["missing_skills"],
            "matching_keywords": mcp_result["matching_keywords"],
            "missing_keywords": mcp_result["missing_keywords"],
            "experience_match": mcp_result["experience_match"],
            "strengths": mcp_result["strengths"],
            "recommendations": mcp_result["recommendations"],
        },
        "mcp": {
            "tool": "analyze_resume_ats",
            "protocol": "Model Context Protocol (MCP) 2024-11-05",
            "status": "success",
            "sample_mode": True
        }
    }


# --------------------------------------------------
# NEW CHAT & DELETE
# --------------------------------------------------

@app.post("/new-chat")
def new_chat():
    global chat_history
    chat_history = []
    return {"message": "New chat session started"}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    global chat_history
    if filename not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_path = documents_store[filename].get("file_path")
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass

    del documents_store[filename]
    chat_history = []
    return {
        "message": "Document removed from session",
        "documents": list(documents_store.keys())
    }

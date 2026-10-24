"""
MCP Server for AI PDF & ATS Assistant.
Exposes Model Context Protocol (MCP) tools for:
- Hybrid PDF RAG Question Answering (with exact page citations)
- Hierarchical Document Summarization
- Live Web Search Grounding (Tavily)
- Resume ATS Scoring, Skills Gap, and Keyword Analysis
- Raw PDF Text Extraction and Available Documents Listing
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Ensure stdout/stderr utf-8 encoding on Windows to prevent stdio corruption
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure silence of noisy telemetry/progress bars on stdio transport
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Base directory
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Load environment
load_dotenv(BASE_DIR / ".env")

from mcp.server.fastmcp import FastMCP

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import get_vector_store
from src.retriever import get_retriever
from src.chatbot import get_llm
from src.rag_chain import ask_question as rag_ask_question
from src.summarizer import summarize_document
from src.web_search import search_web as tavily_search_web
from src.ats.parser import extract_pdf_text
from src.ats.analyzer import analyze_resume

# Initialize FastMCP Server
mcp = FastMCP(
    name="PDF-ATS-Assistant",
    instructions=(
        "Official Model Context Protocol (MCP) server for AI PDF & ATS Assistant. "
        "Provides tools for querying PDFs via semantic & BM25 hybrid search, "
        "document summarization, web search grounding, and ATS resume scoring/gap analysis."
    ),
)

# Global lazy LLM instance and document cache
_LLM = None
_DOC_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_llm_instance():
    global _LLM
    if _LLM is None:
        _LLM = get_llm()
    return _LLM


def _resolve_pdf_path(path_str: str) -> Path:
    """Resolve a PDF path against project root or standard data directories."""
    candidate = Path(path_str)
    if candidate.is_file():
        return candidate.resolve()

    search_roots = [
        BASE_DIR,
        BASE_DIR / "data",
        BASE_DIR / "data" / "ats",
        BASE_DIR / "data" / "uploads",
    ]
    for root in search_roots:
        resolved = (root / path_str).resolve()
        if resolved.is_file():
            return resolved

    # Try matching by filename only
    filename = Path(path_str).name
    for root in search_roots:
        for f in root.glob("*.pdf"):
            if f.name.lower() == filename.lower():
                return f.resolve()

    raise FileNotFoundError(
        f"PDF file '{path_str}' not found in project or data directories."
    )


def _get_or_load_document(pdf_path_str: str) -> Dict[str, Any]:
    """Retrieve or build the hybrid retriever and chunks for a given PDF."""
    resolved_path = _resolve_pdf_path(pdf_path_str)
    cache_key = str(resolved_path)

    if cache_key in _DOC_CACHE:
        return _DOC_CACHE[cache_key]

    # Load and process PDF
    documents = load_pdf(str(resolved_path))
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError(f"Document {resolved_path.name} contains no extractable text.")

    embedding_model = get_embeddings()
    vector_store = get_vector_store(str(resolved_path), chunks, embedding_model)
    retriever = get_retriever(vector_store, chunks)

    _DOC_CACHE[cache_key] = {
        "path": resolved_path,
        "filename": resolved_path.name,
        "chunks": chunks,
        "retriever": retriever,
        "summary": None,
    }
    return _DOC_CACHE[cache_key]


# -----------------------------------------------------------------------------
# MCP Tools
# -----------------------------------------------------------------------------

@mcp.tool()
def list_available_documents() -> List[Dict[str, Any]]:
    """List all available PDF documents across project data, ats, and upload folders.

    Returns:
        A list of document descriptors with filename, relative path, and size in KB.
    """
    found = []
    seen = set()

    search_dirs = [
        ("data", BASE_DIR / "data"),
        ("data/ats", BASE_DIR / "data" / "ats"),
        ("data/uploads", BASE_DIR / "data" / "uploads"),
    ]

    for category, directory in search_dirs:
        if directory.exists():
            for pdf_file in directory.glob("*.pdf"):
                norm = str(pdf_file.resolve()).lower()
                if norm not in seen:
                    seen.add(norm)
                    found.append({
                        "filename": pdf_file.name,
                        "category": category,
                        "relative_path": str(pdf_file.relative_to(BASE_DIR)).replace("\\", "/"),
                        "size_kb": round(pdf_file.stat().st_size / 1024, 1),
                    })

    return found


@mcp.tool()
def extract_pdf_raw_text(pdf_path: str) -> str:
    """Extract raw, plain text from any PDF document.

    Args:
        pdf_path: The filename or path of the PDF document.

    Returns:
        The extracted raw plain text of the document.
    """
    resolved_path = _resolve_pdf_path(pdf_path)
    text = extract_pdf_text(str(resolved_path))
    if not text.strip():
        return f"Warning: No text could be extracted from {resolved_path.name}."
    return text


@mcp.tool()
def ask_pdf(pdf_path: str, question: str) -> str:
    """Answer a question about a PDF document using Hybrid Vector (Chroma) + BM25 search.

    Provides high-precision grounded answers with exact source page citations.

    Args:
        pdf_path: The filename or path of the PDF (e.g. 'AI_Research.pdf' or 'data/AI_Research.pdf').
        question: The user query or question about the document content.

    Returns:
        The grounded answer including page citations if found, or a not-found message.
    """
    doc_info = _get_or_load_document(pdf_path)
    llm = _get_llm_instance()
    retriever = doc_info["retriever"]

    answer = rag_ask_question(
        retriever=retriever,
        llm=llm,
        question=question,
        chat_history=[]
    )
    return answer


@mcp.tool()
def summarize_pdf(pdf_path: str) -> str:
    """Generate a comprehensive, hierarchical executive summary of a PDF document.

    Summarizes chunks and aggregates them into key topics, main ideas, and findings.

    Args:
        pdf_path: The filename or path of the PDF.

    Returns:
        A structured executive summary of the document.
    """
    doc_info = _get_or_load_document(pdf_path)
    if doc_info["summary"] is not None:
        return doc_info["summary"]

    llm = _get_llm_instance()
    summary = summarize_document(doc_info["chunks"], llm)
    doc_info["summary"] = summary
    return summary


@mcp.tool()
def search_web_context(query: str) -> Dict[str, Any]:
    """Perform a live web search using Tavily to verify or supplement information.

    Args:
        query: The search query or question to verify against live web data.

    Returns:
        A dictionary containing the generated answer and a list of cited sources.
    """
    llm = _get_llm_instance()
    result = tavily_search_web(query, llm)
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", [])
    }


@mcp.tool()
def analyze_resume_ats(
    resume_path: str,
    job_description_path: str,
    company_context: str = ""
) -> Dict[str, Any]:
    """Perform a comprehensive ATS resume evaluation against a job description.

    Calculates deterministic ATS match score (0-100), match level, matching & missing
    skills, matching & missing keywords, strengths, and actionable improvement suggestions.

    Args:
        resume_path: Path or filename of the candidate resume PDF.
        job_description_path: Path or filename of the job description PDF.
        company_context: Optional additional company culture or role context.

    Returns:
        A dictionary with overall_score, match_level, matching_skills, missing_skills,
        matching_keywords, missing_keywords, strengths, and recommendations.
    """
    res_path = _resolve_pdf_path(resume_path)
    jd_path = _resolve_pdf_path(job_description_path)

    resume_text = extract_pdf_text(str(res_path))
    jd_text = extract_pdf_text(str(jd_path))

    if not resume_text.strip():
        raise ValueError(f"Could not extract readable text from resume: {res_path.name}")
    if not jd_text.strip():
        raise ValueError(f"Could not extract readable text from job description: {jd_path.name}")

    llm = _get_llm_instance()
    result = analyze_resume(
        resume_text=resume_text,
        job_description=jd_text,
        llm=llm,
        company_context=company_context
    )

    data = result.model_dump()
    return {
        "overall_score": data.get("overall_score"),
        "match_level": data.get("match_level"),
        "matching_skills": data.get("analysis", {}).get("matching_skills", []),
        "missing_skills": data.get("analysis", {}).get("missing_skills", []),
        "matching_keywords": data.get("analysis", {}).get("matching_keywords", []),
        "missing_keywords": data.get("analysis", {}).get("missing_keywords", []),
        "experience_match": data.get("analysis", {}).get("experience_match", ""),
        "strengths": data.get("analysis", {}).get("strengths", []),
        "recommendations": data.get("analysis", {}).get("recommendations", [])
    }


@mcp.tool()
def calculate_ats_score_metric(
    resume_path: str,
    job_description_path: str
) -> Dict[str, Any]:
    """Quickly compute the ATS match score and counts of matching vs missing skills.

    Args:
        resume_path: Path or filename of the resume PDF.
        job_description_path: Path or filename of the job description PDF.

    Returns:
        Summary metric breakdown with score, match level, and counts.
    """
    analysis_data = analyze_resume_ats(resume_path, job_description_path)
    return {
        "overall_score": analysis_data["overall_score"],
        "match_level": analysis_data["match_level"],
        "matching_skills_count": len(analysis_data["matching_skills"]),
        "missing_skills_count": len(analysis_data["missing_skills"]),
        "matching_keywords_count": len(analysis_data["matching_keywords"]),
        "missing_keywords_count": len(analysis_data["missing_keywords"]),
    }


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()

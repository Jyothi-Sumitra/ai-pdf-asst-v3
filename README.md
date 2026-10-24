# 📄 AI PDF Assistant

An AI-powered PDF assistant that allows users to upload documents and interact with them using natural language.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from uploaded PDFs and generate grounded answers with source page citations.

🌐 **Live Demo:** https://ai-pdf-assistant-steel.vercel.app/

---

## ✨ Features

- 📤 Upload and process PDF documents
- 💬 Ask natural-language questions about the uploaded document
- 🔎 Retrieval-Augmented Generation (RAG)
- 🧠 Conversational context for follow-up questions
- 📄 Source page citations in generated answers
- ✨ Whole-document summarization
- 🔍 Hybrid retrieval using semantic search and BM25
- ⚡ Cached vector embeddings for previously processed documents
- 🌐 FastAPI-powered backend
- 🎨 Responsive web interface
- ☁️ Deployed on Render

---

## 🖼️ Preview

![AI PDF Assistant Preview](assets/pdf.png)

---

## 🧠 How It Works

```text
                    PDF Upload
                        │
                        ▼
                  PDF Extraction
                        │
                        ▼
                  Text Chunking
                        │
                        ▼
                Google Embeddings
                        │
                        ▼
                    ChromaDB
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
       Semantic Search        BM25 Search
              │                   │
              └─────────┬─────────┘
                        ▼
                Hybrid Retrieval
                        │
                        ▼
               Relevant Context
                        │
                        ▼
                    Groq LLM
                        │
                        ▼
              Grounded Response
                        │
                        ▼
                 Source Citations
```

When a user uploads a PDF, the document is extracted and split into smaller chunks. Embeddings are generated for those chunks and stored in ChromaDB.

For a question, the application retrieves relevant document chunks using both semantic vector search and BM25 keyword retrieval. The retrieved context is then provided to the language model to generate a document-grounded response.

The application also maintains conversational context so follow-up questions can be understood correctly.

---

## 🛠️ Tech Stack

### AI / RAG

- LangChain
- Groq
- Google Generative AI Embeddings
- ChromaDB
- BM25

### Backend

- Python
- FastAPI
- Uvicorn

### Frontend

- HTML
- CSS
- JavaScript

### Deployment

- Render

---

## 📁 Project Structure

```text
ai-pdf-assistant/
│
├── data/
│
├── src/
│   ├── chatbot.py
│   ├── embeddings.py
│   ├── file_hash.py
│   ├── pdf_loader.py
│   ├── rag_chain.py
│   ├── retriever.py
│   ├── summarizer.py
│   ├── text_splitter.py
│   └── vector_store.py
│
├── static/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   └── favicon.svg
│
├── app.py
├── web_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Jyothi-Sumitra/ai-pdf-assistant.git
cd ai-pdf-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

> Never commit your `.env` file or API keys to GitHub.

### 5. Start the application

```bash
uvicorn web_app:app --reload
```

Then open:

```text
http://127.0.0.1:8000

---

## ☁️ Deploying to Google Cloud Run

This project includes a Dockerfile and CI helpers to deploy to Cloud Run.

Quick local deploy (requires `gcloud` and `docker`):

1. Authenticate and set project:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

2. Export your secrets in the shell (do not check these into Git):

```bash
export PROJECT_ID=your-gcp-project-id
export GROQ_API_KEY="<your_groq_key>"
export GOOGLE_API_KEY="<your_google_key>"
export TAVILY_API_KEY="<your_tavily_key>"
```

3. Run the provided deploy script (uses Cloud Build):

```bash
bash scripts/deploy_cloud_run.sh
```

CI (GitHub Actions):

- Create a service account JSON with roles: `Cloud Run Admin`, `Cloud Build Editor`, `Storage Admin`, `Service Account User`, `Artifact Registry Writer`.
- In your GitHub repo settings -> Secrets, add:
    - `GCP_SA_KEY` (the service account JSON)
    - `GCP_PROJECT` (your project id)
    - `GROQ_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`

Push to `main` to trigger the `.github/workflows/cloud-run.yml` workflow which will build the image and deploy to Cloud Run.

```

---

## 💡 Example Questions

After uploading a PDF, users can ask questions such as:

```text
What is this document about?

Summarize this document.

What are the key topics?

Explain the main ideas in simple terms.

What does the document say about this topic?
```

The assistant retrieves relevant information from the uploaded document and provides an answer based on that context.

---

## 🔐 Security

API keys and other secrets are stored using environment variables and are excluded from version control through `.gitignore`.

Uploaded documents and generated vector database files are also excluded from the repository.

---

## 🌐 Live Demo

Try the deployed application:

**https://ai-pdf-assistant-cdly.onrender.com/**

> The application is hosted on Render's free tier, so the first request after a period of inactivity may take additional time while the service starts.

---


---

## ⚡ Model Context Protocol (MCP) Integration

This project natively supports Anthropic's open **Model Context Protocol (MCP)**, exposing all PDF RAG, summarization, web search, and ATS resume scoring capabilities as standard tools for MCP clients (such as **Claude Desktop**, **Cursor**, or custom autonomous agents).

### 🛠️ Registered MCP Tools

| Tool | Description |
|------|-------------|
| `ask_pdf(pdf_path, question)` | Hybrid Semantic (Chroma) + BM25 retrieval over any PDF with source page citations |
| `summarize_pdf(pdf_path)` | Hierarchical executive summarization of multi-page PDF documents |
| `search_web_context(query)` | Live web search grounding using Tavily to verify or supplement context |
| `analyze_resume_ats(resume_path, job_description_path)` | Deterministic ATS score (0-100), match level, skills gap analysis, and tailored recommendations |
| `calculate_ats_score_metric(resume_path, job_description_path)` | Quick metric breakdown of matching vs. missing skills and keywords |
| `extract_pdf_raw_text(pdf_path)` | Extract raw readable text from any PDF document |
| `list_available_documents()` | Inspect all available PDFs in repository `data/`, `data/ats/`, and `data/uploads/` folders |

### 🚀 Running the MCP Server

Start the FastMCP server over standard I/O:
```bash
python mcp_server.py
```

### 🤖 Autonomous MCP ReAct Agent

An autonomous agent (`mcp_agent.py`) connects to the server via stdio JSON-RPC, dynamically discovers all registered tools, and executes multi-step reasoning:

```bash
# Automated ATS and document demonstration
python mcp_agent.py --demo

# Run an ad-hoc query
python mcp_agent.py --query "Summarize data/AI_Research.pdf and identify key research challenges"

# Interactive CLI mode
python mcp_agent.py
```

### 🧪 Running the MCP Test Suite

Validate MCP protocol handshake, tool discovery, and execution across all 7 tools:
```bash
python test_mcp.py
```

### 🔌 Claude Desktop Configuration

To use this assistant inside **Claude Desktop**, add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pdf-ats-assistant": {
      "command": "C:\\Users\\SUMITHRA\\Downloads\\Work\\Python\\AI-Projects\\PDF Assistant\\.venv\\Scripts\\python.exe",
      "args": [
        "C:\\Users\\SUMITHRA\\Downloads\\Work\\Python\\AI-Projects\\PDF Assistant\\mcp_server.py"
      ],
      "env": {
        "GROQ_API_KEY": "your_groq_api_key_here",
        "GOOGLE_API_KEY": "your_google_api_key_here",
        "TAVILY_API_KEY": "your_tavily_api_key_here"
      }
    }
  }
}
```

> **Note on Web UI:** The original web interface (`uvicorn web_app:app` serving `static/index.html` and `static/ats.html`) remains 100% intact, preserved, and operational.

## 📌 Future Improvements

Potential future improvements include:

- Support for multiple documents
- Streaming responses
- Improved reranking and retrieval evaluation
- Persistent user sessions
- Support for additional document formats

---

## 👩‍💻 Author

Built as a hands-on Generative AI project exploring **RAG, embeddings, vector databases, hybrid retrieval, conversational AI, FastAPI, and LLM application deployment**.

from langchain_chroma import Chroma
import os

from src.file_hash import calculate_file_hash


DB_ROOT = os.getenv(
    "CHROMA_DB_ROOT",
    "/tmp/pdf-assistant/chroma_db" if os.getenv("VERCEL") else "chroma_db"
)


def get_vector_store(pdf_path, chunks, embedding_model):

    # Unique ID for this PDF
    file_hash = calculate_file_hash(pdf_path)

    # Each PDF gets its own vector database
    db_path = os.path.join(
        DB_ROOT,
        file_hash
    )

    # PDF already processed
    if os.path.exists(db_path):

        print(f"📂 Loading existing vector database: {pdf_path}")

        return Chroma(
            persist_directory=db_path,
            embedding_function=embedding_model
        )

    # New PDF
    print(f"🆕 Creating vector database: {pdf_path}")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path
    )

    return vector_store
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import hashlib
from dotenv import load_dotenv

load_dotenv()


class LocalHashEmbeddings:
    def _embed(self, text):
        values = []
        seed = text.encode("utf-8", errors="ignore")

        for index in range(12):
            digest = hashlib.sha256(seed + index.to_bytes(2, "big")).digest()
            values.extend(byte / 255 for byte in digest)

        return values

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def get_embeddings():
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        return LocalHashEmbeddings()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key
    )
    return embeddings
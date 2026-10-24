from langchain_community.retrievers import BM25Retriever


class HybridRetriever:
    def __init__(self, vector_store, chunks):
        if not chunks:
            raise ValueError(
                "Cannot create a retriever from an empty document."
            )

        # Semantic search
        self.vector_retriever = (
            vector_store.as_retriever(search_kwargs={"k": 4})
            if vector_store is not None
            else None
        )

        # Keyword search
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 4

    def invoke(self, query):
        # Get results from both retrievers
        vector_docs = (
            self.vector_retriever.invoke(query)
            if self.vector_retriever is not None
            else []
        )
        bm25_docs = self.bm25_retriever.invoke(query)

        # Combine results while avoiding duplicates
        combined_docs = []
        seen = set()

        for doc in bm25_docs + vector_docs:
            key = (
                doc.page_content,
                doc.metadata.get("page")
            )

            if key not in seen:
                seen.add(key)
                combined_docs.append(doc)

        return combined_docs


def get_retriever(vector_store, chunks):
    return HybridRetriever(vector_store, chunks)
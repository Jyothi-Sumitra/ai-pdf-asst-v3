
from langchain_core.prompts import ChatPromptTemplate

NOT_FOUND = "I couldn't find that information in the document."

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the user's question using ONLY the context below.
Do not use outside knowledge or invent information.

If the answer is not found in the context, say exactly:
"I couldn't find that information in the document."

Context:
{context}

Question:
{question}
"""
)

history_prompt = ChatPromptTemplate.from_template(
    """
Rewrite the latest user question ONLY if it depends on
the conversation history.

Rules:
- If the question is already understandable on its own,
  return it EXACTLY as written.
- Use history only to resolve missing or ambiguous references.
- Preserve the user's intent.
- Never carry a previous question's intent into a new,
  standalone question.
- Do not answer the question.
- Output only the final question.

Conversation history:
{chat_history}

Latest question:
{question}
"""
)


def ask_question(retriever, llm, question, chat_history):
    # 1. Rewrite the question only when history is available.
    standalone_question = question

    if chat_history:
        history_messages = history_prompt.format_messages(
            chat_history=chat_history,
            question=question,
        )

        rewritten_response = llm.invoke(history_messages)
        standalone_question = (
            rewritten_response.content.strip() or question
        )

    # 2. Retrieve relevant document chunks.
    docs = retriever.invoke(standalone_question)

    # 3. Avoid generating an answer without document evidence.
    if not docs:
        return NOT_FOUND

    context = "\n\n".join(doc.page_content for doc in docs)

    if not context.strip():
        return NOT_FOUND

    # 4. Answer the same question used for retrieval.
    messages = prompt.format_messages(
        context=context,
        question=standalone_question,
    )

    response = llm.invoke(messages)
    answer = response.content.strip()

    # 5. Preserve the existing not-found behavior.
    if NOT_FOUND.lower() in answer.lower():
        return NOT_FOUND

    # 6. Attach source page numbers.
    pages = sorted(
        {
            doc.metadata["page"] + 1
            for doc in docs
            if isinstance(doc.metadata.get("page"), int)
        }
    )

    if pages:
        sources = ", ".join(str(page) for page in pages)
        return f"{answer}\n\n📄 Sources: Page(s) {sources}"

    return answer
